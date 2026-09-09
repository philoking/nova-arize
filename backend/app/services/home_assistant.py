"""Home Assistant client + tool definitions for the assistant's agent loop.

**Curated allowlist model.** The assistant may see and control ONLY the entities
carrying the configured HA label (``NOVA_VOICE_HA_LABEL``, default ``nova``) - a
small set the user opts in to, instead of the whole ~2,000-entity registry. This
is a safety measure: with open-ended fuzzy search the model toggled the *wrong*
device (fireplace light instead of the kitchen island) because they shared words.

The labeled set (entity_id, friendly name, area, domain, state) is fetched in one
HA *template* call, cached, and:
- injected into the system prompt (via the provider's ``system_note``) so the
  model references exact names/ids,
- searched by ``find_entities`` - over the curated set only, never the registry,
- enforced by ``get_state`` / ``call_service``, which refuse any entity_id not on
  the allowlist (defense in depth, on top of the control-domain guard).

The long-lived token lives only here, server-side.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import httpx

from ..config import settings

# Filler words dropped from a search query before matching. Nouns/adjectives
# (including "light") are kept - they're what actually discriminate entities.
_STOPWORDS = {"the", "a", "an", "to", "my", "in", "of", "and", "turn", "on",
              "off", "please", "set", "is", "are", "what", "s"}


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens, minus stopwords, with naive plural stemming."""
    out = set()
    for raw in re.split(r"[^a-z0-9]+", text.lower()):
        if len(raw) < 2 or raw in _STOPWORDS:
            continue
        out.add(raw[:-1] if len(raw) > 3 and raw.endswith("s") else raw)
    return out


# Attribute keys worth showing the model; the full attribute blob is noisy.
_USEFUL_ATTRS = (
    "friendly_name",
    "unit_of_measurement",
    "device_class",
    "brightness",
    "color_temp_kelvin",
    "rgb_color",
    "current_temperature",
    "temperature",
    "hvac_action",
    "hvac_modes",
    "preset_mode",
)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.ha_url,
        headers={"Authorization": f"Bearer {settings.ha_token}"},
        timeout=15,
        verify=settings.ha_verify_ssl,
    )


def _slim(entity: dict) -> dict:
    """Reduce a full HA state object to the fields the model needs."""
    attrs = entity.get("attributes", {})
    out = {
        "entity_id": entity.get("entity_id"),
        "name": attrs.get("friendly_name", entity.get("entity_id")),
        "state": entity.get("state"),
    }
    extra = {k: attrs[k] for k in _USEFUL_ATTRS if k in attrs and k != "friendly_name"}
    if extra:
        out["attributes"] = extra
    return out


# Curated device registry (the labeled allowlist)
# Cached snapshot of label_entities(<label>) with name/area/domain/state. Assigned
# atomically on refresh; readers see a consistent list without a lock.
_devices: list[dict] = []
_fetched_at: float = 0.0


def _devices_template(label: str) -> str:
    """A Jinja template that returns the labeled entities as a JSON array of
    {entity_id, name, area, domain, state}. One round-trip for the whole set."""
    safe = re.sub(r"[^a-z0-9_]", "", label.lower()) or "nova"
    return (
        "{% set ns = namespace(items=[]) %}"
        "{% for e in label_entities('" + safe + "') %}"
        "{% set ns.items = ns.items + [{"
        "'entity_id': e, "
        "'name': state_attr(e, 'friendly_name') or e, "
        "'area': area_name(e), "
        "'domain': e.split('.')[0], "
        "'state': states(e)}] %}"
        "{% endfor %}"
        "{{ ns.items | tojson }}"
    )


async def refresh_devices() -> list[dict]:
    """Re-fetch the labeled device set from HA and update the cache. On any error
    the previous cache is kept (a transient HA blip shouldn't empty the allowlist)."""
    global _devices, _fetched_at
    if not settings.ha_enabled:
        return []
    try:
        async with _client() as client:
            resp = await client.post("/api/template", json={"template": _devices_template(settings.ha_label)})
            resp.raise_for_status()
            items = json.loads(resp.text)
    except (httpx.HTTPError, ValueError):
        _fetched_at = time.monotonic()  # don't hammer HA on a persistent error
        return _devices
    if isinstance(items, list):
        _devices = sorted(items, key=lambda d: ((d.get("area") or "~"), d.get("name") or ""))
    _fetched_at = time.monotonic()
    return _devices


def devices() -> list[dict]:
    """The cached curated device list (may be up to ``ha_devices_ttl`` stale)."""
    return _devices


def allowlist() -> set[str]:
    return {d["entity_id"] for d in _devices}


async def ensure_fresh() -> None:
    """Refresh the cache if it's older than the TTL (or never loaded)."""
    if not _devices or (time.monotonic() - _fetched_at) > settings.ha_devices_ttl:
        await refresh_devices()


async def refresh_loop() -> None:
    """Keep the cache warm so the injected system prompt stays current even
    between tool calls. Started at app startup when HA is enabled."""
    import asyncio
    while True:
        await asyncio.sleep(max(30, settings.ha_devices_ttl))
        try:
            await refresh_devices()
        except Exception:  # noqa: BLE001 - long-running daemon loop
            pass


# Tools

async def find_entities(query: str, limit: int = 15) -> dict:
    """Search the *curated* device list by name/area - never the full registry.

    Ranks the labeled devices by how many of the query's words hit the device's
    name + area. Returns the matches (or, if nothing matches, the whole small
    list) so the model can pick the exact entity_id. Safe by construction: only
    labeled devices are ever returned.
    """
    await ensure_fresh()
    q_tokens = _tokens(query or "")
    scored = []
    for d in _devices:
        hay = _tokens(f"{d.get('name', '')} {d.get('area') or ''} {d.get('entity_id', '')}")
        score = len(q_tokens & hay)
        if query and query.strip().lower() in (d.get("name", "") or "").lower():
            score += len(q_tokens) + 1  # verbatim name hit → strong
        if score or not q_tokens:
            scored.append((score, d))
    scored.sort(key=lambda s: (-s[0], s[1].get("entity_id", "")))
    matches = [d for _, d in scored[:limit]]
    # No token overlap at all → hand back the full (small) curated list to choose from.
    if not matches:
        matches = _devices[:limit]
    return {"count": len(matches), "devices": matches}


async def get_state(entity_id: str) -> dict:
    await ensure_fresh()
    if entity_id not in allowlist():
        return {"error": f"'{entity_id}' is not one of Nova's devices; only use the listed devices."}
    async with _client() as client:
        resp = await client.get(f"/api/states/{entity_id}")
        if resp.status_code == 404:
            return {"error": f"unknown entity_id: {entity_id}"}
        resp.raise_for_status()
        return _slim(resp.json())


async def _settle(client: httpx.AsyncClient, entity_id: str, want: str | None,
                  tries: int = 6, delay: float = 0.3) -> dict | None:
    """Poll an entity's state until it reaches ``want`` (or a few tries elapse).

    HA's immediate service response is unreliable for groups/bulbs (they update
    asynchronously and the response often lists no changes), so we read the real
    resulting state instead of trusting the call returned. Returns the raw state
    dict, or None if it couldn't be read.
    """
    state = None
    for _ in range(tries):
        await asyncio.sleep(delay)
        r = await client.get(f"/api/states/{entity_id}")
        if r.status_code == 200:
            state = r.json()
            if want is None or state.get("state") == want:
                return state
    return state


async def call_service(domain: str, service: str, entity_id: str, data: dict | None = None) -> dict:
    """Call a HA service - only on an allowlisted entity, in a permitted domain -
    then verify the real resulting state instead of assuming success.

    For on/off, we poll until the device reports the intended state and retry the
    call once if it didn't take (a group member can miss a command issued right
    after another). If it still hasn't, we report the actual state and flag it, so
    the model tells the user the truth rather than a false "done".
    """
    await ensure_fresh()
    if entity_id not in allowlist():
        return {"error": (f"'{entity_id}' is not one of Nova's controllable devices. Only control "
                          "devices from the provided list; if unsure which the user means, ask.")}
    allowed = settings.ha_control_domain_set
    if domain not in allowed:
        return {"error": f"control of '{domain}' is not enabled; allowed domains: {sorted(allowed)}"}
    payload = {"entity_id": entity_id, **(data or {})}
    # Services with a known resulting state get the settle-and-verify treatment
    # below, so we report the real outcome instead of trusting HA's response.
    want = {"turn_on": "on", "turn_off": "off",
            "lock": "locked", "unlock": "unlocked"}.get(service)

    async with _client() as client:
        state = None
        for _ in range(2):  # initial call, plus one retry if an on/off didn't take
            resp = await client.post(f"/api/services/{domain}/{service}", json=payload)
            if resp.status_code >= 400:
                return {"error": f"HA returned {resp.status_code}: {resp.text[:200]}"}
            state = await _settle(client, entity_id, want)
            if want is None or (state and state.get("state") == want):
                break

    result = _slim(state) if state else "service called"
    if want and state and state.get("state") != want:
        # Be honest: the command was accepted but the device isn't in the intended
        # state, so the model shouldn't claim it succeeded.
        return {"ok": False, "entity_id": entity_id, "result": result,
                "note": f"the device still reports '{state.get('state')}' — it may not have responded"}
    return {"ok": True, "entity_id": entity_id, "result": result}


# Tool specs (Ollama / OpenAI function-calling format)

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "find_entities",
            "description": (
                "Search Nova's controllable devices (a small curated list, also given in the system "
                "prompt) by a spoken name or keyword, e.g. 'kitchen lights' or 'thermostat'. Returns "
                "the matching devices with their exact entity_id and current state. Only these devices "
                "exist to Nova — never reference an entity_id that isn't among them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Name/keyword, e.g. 'office' or 'kitchen light'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "Read the current state and attributes of one of Nova's devices by its entity_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "an entity_id from the device list"},
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_service",
            "description": (
                "Control one of Nova's devices via a Home Assistant service. Examples: turn a light "
                "on/off (domain 'light', service 'turn_on'/'turn_off'; brightness via data "
                "{'brightness_pct': 30}), set a thermostat (domain 'climate', service "
                "'set_temperature', data {'temperature': 70}), or lock/unlock a door (domain 'lock', "
                "service 'lock'/'unlock'). The entity_id must be one from the "
                "device list; anything else is refused."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "e.g. 'light' or 'climate'"},
                    "service": {"type": "string", "description": "e.g. 'turn_on', 'turn_off', 'set_temperature'"},
                    "entity_id": {"type": "string", "description": "target entity id (from the device list)"},
                    "data": {"type": "object", "description": "extra service data, e.g. {'brightness_pct': 30}"},
                },
                "required": ["domain", "service", "entity_id"],
            },
        },
    },
]

_DISPATCH = {
    "find_entities": find_entities,
    "get_state": get_state,
    "call_service": call_service,
}


async def execute_tool(name: str, arguments: dict) -> dict:
    """Run a tool call by name. Never raises - errors come back as data so the
    model can react (retry, clarify, or explain)."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return await fn(**(arguments or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except httpx.HTTPError as exc:
        return {"error": f"home assistant request failed: {exc}"}


async def health() -> bool:
    if not settings.ha_enabled:
        return False
    try:
        async with _client() as client:
            resp = await client.get("/api/")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
