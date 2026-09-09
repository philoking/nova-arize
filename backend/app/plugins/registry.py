"""Aggregates enabled providers into a single tool surface for the agent loop.

The registry owns *namespacing*: each provider's bare tool names get a
``<provider>__`` prefix on the way out (so the model sees globally-unique names
that still satisfy the ``[A-Za-z0-9_-]+`` constraint Ollama/OpenAI impose - a
dot would be rejected, hence ``__``), and the prefix is stripped again before a
call is dispatched back to the owning provider.
"""

from __future__ import annotations

import asyncio
import copy
import re

from .. import skills as _skills
from .base import ToolProvider

SEP = "__"

# qwen3 is a thinking model. We pass think: false and strip <think> blocks, but it
# still leaks control tokens into tool-call string arguments (content="/no_think"),
# which would be written verbatim into a note or list (#19). Scrub in dispatch so
# every provider is covered.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TOKEN = re.compile(r"\s*/(?:no_)?think\b", re.IGNORECASE)  # /no_think or /think


def _scrub(value):
    """Recursively strip qwen3 control tokens from string args (dicts/lists too)."""
    if isinstance(value, str):
        out = _THINK_TOKEN.sub("", _THINK_BLOCK.sub("", value))
        if out == value:
            return value  # nothing removed - leave the value byte-for-byte intact
        return re.sub(r"[ \t]{2,}", " ", out).strip()
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


class Registry:
    def __init__(self, providers: list[ToolProvider]):
        # Keep only configured providers; a disabled one contributes nothing.
        self._providers = [p for p in providers if p.enabled]
        self._by_name = {p.name: p for p in self._providers}

    @property
    def providers(self) -> list[ToolProvider]:
        return list(self._providers)

    def tools(self) -> list[dict] | None:
        """Merged, namespaced tool specs - or ``None`` when nothing is enabled.

        Returning ``None`` (rather than ``[]``) lets the LLM layer skip the
        tool-calling machinery entirely and stream a plain reply.
        """
        specs: list[dict] = []
        for p in self._providers:
            for spec in p.tool_specs():
                spec = copy.deepcopy(spec)  # don't mutate the provider's constant
                fn = spec["function"]
                fn["name"] = f"{p.name}{SEP}{fn['name']}"
                specs.append(spec)
        return specs or None

    async def execute(self, tool: str, args: dict) -> dict:
        """Dispatch a namespaced tool call to the provider that owns it."""
        ns, sep, bare = tool.partition(SEP)
        if not sep:
            return {"error": f"un-namespaced tool call: {tool!r}"}
        provider = self._by_name.get(ns)
        if provider is None:
            return {"error": f"unknown tool namespace: {ns!r}"}
        return await provider.execute(bare, _scrub(args or {}))

    def system_note(self) -> str:
        """Concatenated system-prompt fragments from every enabled provider.

        Kept for callers that want the full note; the routed chat path uses
        ``context_for`` (only the selected skills' fragments) instead.
        """
        return "".join(p.system_note() for p in self._providers)

    # Capability routing (#35/#39)
    # Pick the 1-2 skills a turn needs and expose only their tools + notes, so a
    # small model isn't choosing among the whole (growing) tool catalog.

    def _available_tool_names(self) -> set[str]:
        return {s["function"]["name"] for s in (self.tools() or [])}

    def _ha_device_keywords(self) -> list[str]:
        """Labeled HA device names as extra deterministic triggers for the 'home'
        skill, so 'turn on the office lamp' routes even without a generic keyword."""
        try:
            from ..services import home_assistant as ha
            out = []
            for d in ha.devices():
                n = "".join(c if c.isalnum() or c.isspace() else " " for c in (d.get("name") or "").lower())
                n = " ".join(n.split())
                if n:
                    out.append(n)
            return out
        except Exception:  # noqa: BLE001 - routing must never fail the turn
            return []

    # Words in a manual's title that don't identify the equipment - brand/model
    # tokens and the equipment noun are what a user names, not these.
    _MANUAL_STOPWORDS = frozenset({
        "manual", "manuals", "user", "users", "guide", "owner", "owners", "instruction",
        "instructions", "operator", "operators", "handbook", "reference", "pdf", "the",
        "and", "for", "with", "your",
    })

    def _manual_keywords(self) -> list[str]:
        """Uploaded manual titles (full phrase + each significant word) as extra
        triggers for the 'manuals' skill, so 'what blade does my grizzly table saw
        use' routes on 'grizzly'/'table' even without a 'manual'/'spec' keyword -
        the same trick as HA device names. People name their *equipment*, not the
        word 'manual'."""
        try:
            from .. import manuals as _manuals
            out: list[str] = []
            for m in _manuals.list_all():
                if m.get("status") != "ready":
                    continue
                title = "".join(c if c.isalnum() or c.isspace() else " " for c in (m.get("title") or "").lower())
                title = " ".join(title.split())
                if not title:
                    continue
                out.append(title)  # the full title as a phrase
                for w in title.split():  # plus each identifying word (brand/model/noun)
                    if len(w) >= 3 and w not in self._MANUAL_STOPWORDS:
                        out.append(w)
            return list(dict.fromkeys(out))  # de-dup, keep order
        except Exception:  # noqa: BLE001 - routing must never fail the turn
            return []

    # Position/orientation words that appear in many camera names but don't identify
    # a place on their own - kept only inside the full-name phrase, never as a lone
    # trigger (so 'turn on the back light' doesn't route to cameras).
    _CAMERA_GENERIC = frozenset({
        "back", "front", "main", "inside", "outside", "side", "room", "yard", "upper",
        "lower", "rear", "house", "camera", "cam", "area", "north", "south", "east", "west",
    })

    def _camera_keywords(self) -> list[str]:
        """Frigate camera friendly names (full phrase + each *distinctive* word) as
        extra triggers for the 'cameras' skill, so 'pull up the driveway' or 'show
        me the chicken coop' route even without the word 'camera' - the same trick
        as HA device names. People name the *place*, not 'camera'."""
        try:
            from ..services import frigate
            out: list[str] = []
            for c in frigate.cameras():
                name = (c.get("name") or "").lower().strip()
                if not name:
                    continue
                out.append(name)  # the full name as a phrase ("chicken coop inside")
                for w in name.split():  # plus each place word ("driveway", "porch")
                    if len(w) >= 4 and w not in self._CAMERA_GENERIC:
                        out.append(w)
            return list(dict.fromkeys(out))  # de-dup, keep order
        except Exception:  # noqa: BLE001 - routing must never fail the turn
            return []

    def route(self, user_texts: list[str]) -> list[str]:
        """Skill names for this turn by deterministic keyword/intent match (no model
        call - zero added latency). ``user_texts`` is the recent user messages,
        oldest→newest. Empty match ⇒ the turn runs tool-free (plain conversation).

        A follow-up that references the last request with a pronoun/continuation
        word ('turn *them* off', '*they* are still on') inherits the skill of the
        most recent prior turn that matched, so pronoun references still reach the
        right tool. Broaden a skill's ``keywords`` in ``skills.py`` if a phrasing
        isn't caught."""
        active = _skills.available(self._available_tool_names())
        current = (user_texts[-1] if user_texts else "").strip()
        if not active or not current:
            return []
        extra = {"home": self._ha_device_keywords(), "manuals": self._manual_keywords(),
                 "cameras": self._camera_keywords()}
        hit = _skills.deterministic_match(current, active, extra)
        # Math is expressed with operator words/symbols, not tidy keywords, so add
        # the 'math' skill by heuristic. Additive: a false positive only exposes the
        # (unused) calculate tool, never hides another skill's tools.
        if any(s.name == "math" for s in active) and "math" not in hit and _skills.looks_like_math(current):
            hit = hit + ["math"]
        hit = self._add_manual_web_fallback(hit, active)
        if hit:
            return hit
        if _skills.is_followup(current):
            for prior in reversed(user_texts[:-1]):
                prior_hit = _skills.deterministic_match(prior, active, extra)
                if prior_hit:
                    # The follow-up ("it's a Rikon 10-325") must keep web_search too,
                    # or a manual-miss escalation can't complete on the next turn.
                    return self._add_manual_web_fallback(prior_hit, active)
        return []

    def _add_manual_web_fallback(self, hit: list[str], active: list) -> list[str]:
        """When the 'manuals' skill routes, also expose web_search (if enabled) so a
        manual-miss can fall back to the web instead of guessing. Additive; the
        manuals note tells the model to try the manual first and attribute the web
        part clearly."""
        if "manuals" in hit and "web" not in hit and any(s.name == "web" for s in active):
            return hit + ["web"]
        return hit

    def tools_for(self, skill_names: list[str]) -> list[dict] | None:
        """The merged tool specs for the selected skills (or ``None`` if none)."""
        if not skill_names:
            return None
        wanted: set[str] = set()
        for name in skill_names:
            skill = _skills._BY_NAME.get(name)
            if skill:
                wanted.update(skill.tools)
        specs = [s for s in (self.tools() or []) if s["function"]["name"] in wanted]
        return specs or None

    def context_for(self, skill_names: list[str]) -> str:
        """Always-on context (time, memory, ringing alarm) plus the selected
        skills' prompt fragments - a small fraction of the full system note."""
        note = _skills.always_on_context()
        for name in dict.fromkeys(skill_names):  # de-dup, keep order
            skill = _skills._BY_NAME.get(name)
            if skill is None:
                continue
            if skill.provider_note:
                prov = self._by_name.get(skill.provider_note)
                if prov is not None:
                    note += prov.system_note()
            else:
                note += skill.note
        return note

    async def health(self) -> dict[str, bool]:
        """Reachability of each enabled provider, keyed by provider name."""
        if not self._providers:
            return {}
        results = await asyncio.gather(*(p.health() for p in self._providers))
        return {p.name: ok for p, ok in zip(self._providers, results)}
