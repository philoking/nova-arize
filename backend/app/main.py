"""Nova Voice backend - a backend-for-frontend for a private voice assistant.

The browser talks only to this service (same origin), which proxies to the three
private AI services on Nova server-side:

    mic ─▶ /api/stt  ─▶ ceres-stt (whisper)   ─▶ transcript
           /api/chat ─▶ ollama (qwen3)         ─▶ streamed reply
           /api/tts  ─▶ kokoro                 ─▶ streamed speech ─▶ speaker

Keeping the AI services off the public internet and out of the browser avoids
CORS and mixed-content problems and keeps everything on the LAN.
"""

from __future__ import annotations

import json
import os
import re

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, dashboard, escalation, history, logstore, manuals, memory, satellites, skills as _skills, timers, tracing
from .config import settings
from .logstore import log
from .plugins import Registry
from .plugins.calculator import CalculatorProvider
from .plugins.cameras import CamerasProvider
from .plugins.home_assistant import HomeAssistantProvider
from .plugins.manuals import ManualsProvider
from .plugins.memory import MemoryProvider
from .plugins.obsidian import ObsidianProvider
from .plugins.timers import TimersProvider, format_timer
from .plugins.websearch import WebSearchProvider
from .plugins.registry import SEP as TOOL_NS_SEP
from .services import claude, frigate as frigate_svc, llm, metrics as metrics_svc, stt, tts
from .settings_store import store

app = FastAPI(title=f"{settings.assistant_name} Assistant", version=__version__)

# The capability registry. Append new providers here - nothing else in this file
# or in the LLM layer needs to change when a capability is added.
registry = Registry([
    HomeAssistantProvider(),
    ObsidianProvider(),
    TimersProvider(),
    WebSearchProvider(),
    MemoryProvider(),
    CalculatorProvider(),
    ManualsProvider(),
    CamerasProvider(),
])


@app.on_event("startup")
async def _startup() -> None:
    # Wire the Arize exporter before anything else runs, so startup work is
    # traceable too. No-op unless ARIZE_SPACE_ID + ARIZE_API_KEY are set.
    tracing.init()
    # Capture the app's log records into the SQLite ring buffer exposed at
    # /api/log (the Settings-screen viewer), and drop anything past the window.
    logstore.init()
    logstore.install()
    logstore.reclassify_noise()  # re-tag existing rows against the current noise rules (#55)
    logstore.prune()
    # Open the shared DB and drop anything past the rolling window. The DALs also
    # lazily init on first use, so this is just eager setup + housekeeping.
    history.init()
    history.prune()
    timers.init()
    timers.prune()
    memory.init()
    memory.prune()
    manuals.init()
    satellites.init()
    log.info(
        "%s backend started — model=%s ha=%s obsidian=%s websearch=%s escalation=%s",
        settings.assistant_name, settings.model, settings.ha_enabled,
        settings.obsidian_enabled, settings.websearch_enabled, settings.claude_enabled,
    )
    # Background loop that marks timers 'fired' at their time. Delivery is by
    # polling from the web panel and the satellite; the on_fire hook additionally
    # emits a phone push over MQTT (#16) when notifications are enabled.
    import asyncio
    from .services import notify

    def _on_timer_fire(t):
        # Fire-and-forget: a broker that's down/unset must not stall the scheduler.
        asyncio.create_task(notify.emit_timer(t))

    asyncio.create_task(timers.run_scheduler(on_fire=_on_timer_fire))
    # Background manual-ingest worker (#63): indexes uploaded PDFs one at a time so
    # a bulk import returns fast and the heavy embed/index work trickles through.
    # Re-enqueue anything a prior run left unfinished before the worker starts.
    manuals.requeue_pending()
    asyncio.create_task(manuals.run_worker())
    # Warm the curated HA device list (the allowlist injected into the prompt),
    # then keep it fresh so re-labeling in HA shows up without a restart.
    if settings.ha_enabled:
        from .services import home_assistant as ha
        await ha.refresh_devices()
        asyncio.create_task(ha.refresh_loop())
    # Warm the Frigate camera catalog (injected into the prompt + used to resolve
    # 'pull up the driveway'), then keep it fresh as cameras are added/removed.
    if settings.frigate_enabled:
        from .services import frigate
        await frigate.refresh_cameras()
        asyncio.create_task(frigate.refresh_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    # Flush buffered spans before the process dies. Without this, every deploy
    # silently drops whatever the batch processor was still holding.
    tracing.shutdown()


@app.middleware("http")
async def _revalidate_frontend(request: Request, call_next):
    """Make the browser revalidate the frontend every load instead of serving it
    stale from a heuristic cache.

    StaticFiles sends only ETag/Last-Modified (no Cache-Control), so browsers
    heuristically cache the shell and a deploy wouldn't show up without a hard
    refresh - a bad fit for a same-origin LAN app. ``no-cache`` means "store it,
    but always revalidate first": the ETag makes that a cheap 304 when nothing
    changed, and fresh bytes the moment it does. API responses are dynamic and
    never cached, so this only matters for the static files.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# API

@app.get("/api/health")
async def health() -> JSONResponse:
    import asyncio

    # Tag the upstream probes this fans out as noise so the log viewer can hide
    # the routine health chatter (the flag propagates into the gathered tasks).
    token = logstore.health_check_var.set(True)
    try:
        (stt_ok, llm_ok, tts_ok), plugin_health = await asyncio.gather(
            asyncio.gather(stt.health(), llm.health(), tts.health()),
            registry.health(),
        )
    finally:
        logstore.health_check_var.reset(token)
    services = {"stt": stt_ok, "llm": llm_ok, "tts": tts_ok, **plugin_health}
    # Plugins are optional; the core voice loop is healthy without them.
    ok = stt_ok and llm_ok and tts_ok
    return JSONResponse({"ok": ok, "services": services}, status_code=200 if ok else 503)


@app.get("/api/config")
async def config() -> dict:
    """Bootstrap data for the frontend: model, default voice, and voice list."""
    try:
        voices = await tts.list_voices()
    except httpx.HTTPError:
        voices = [settings.voice]
    return {
        "name": settings.assistant_name,
        "model": store.get("model"),
        # The effective voice = user's saved choice, else the env default. Both
        # the browser and the satellite fall back to this when they send no voice.
        "voice": store.get("voice"),
        "voices": voices,
        # Enabled capability namespaces, for the frontend to reflect/label.
        "capabilities": [p.name for p in registry.providers],
        "home_assistant": settings.ha_enabled,  # kept for back-compat
        # Whether opt-in Claude escalation is available (drives the UI affordance).
        "escalation": settings.claude_enabled,
        # Whether live homelab metrics are available (drives the constellation HUD).
        "metrics": settings.metrics_enabled,
        # The two escalation tiers, so the constellation can label + light the right
        # cloud node: `fast` (reactive struggle → Haiku) vs `deep` (upfront
        # research/writing → Opus).
        "escalation_models": {"fast": settings.claude_fast_model, "deep": settings.claude_model},
        # Rolling window (days) the conversation history is kept for.
        "history_days": settings.history_days,
    }


# CORE capabilities have no external dependency (always available); every other
# provider is config-gated (needs a token/service), so its cluster renders muted
# until connected. Keyed by provider namespace.
_CORE_PROVIDERS = {"timers", "memory", "calculator"}


@app.get("/api/capabilities")
async def capabilities() -> dict:
    """The capability graph the constellation UI renders.

    Each routing skill (see ``skills.py``) is a cluster: its backing provider,
    whether that provider is currently enabled, whether it's CORE or config-gated,
    and its tool leaves (bare names). Derived live from ``skills.SKILLS`` + the
    registry, so adding a Skill/provider makes it appear with no frontend change.
    The frontend owns only the *layout* (positions/colours); the *contents* here
    stay authoritative about which tools actually exist. Connected/disconnected
    state comes from ``/api/health`` (keyed by the same ``provider`` name).
    """
    enabled_names = {s["function"]["name"] for s in (registry.tools() or [])}
    skills_out = []
    for skill in _skills.SKILLS:
        if not skill.tools:
            continue
        provider = skill.tools[0].split(TOOL_NS_SEP, 1)[0]
        skills_out.append({
            "name": skill.name,
            "provider": provider,
            "kind": "CORE" if provider in _CORE_PROVIDERS else "GATED",
            "enabled": any(t in enabled_names for t in skill.tools),
            "functions": [t.split(TOOL_NS_SEP, 1)[-1] for t in skill.tools],
        })
    return {"model": store.get("model"), "skills": skills_out}


@app.get("/api/settings")
async def get_settings() -> dict:
    """The user-editable settings and the options for each.

    Backs the Settings screen: ``settings`` is the current effective value of
    each writable key, ``options`` the allowed values (e.g. the TTS voice list).
    """
    try:
        voices = await tts.list_voices()
    except httpx.HTTPError:
        voices = [store.get("voice")]
    models = await llm.list_models() or [store.get("model")]
    return {
        "settings": store.all(),
        "options": {"voice": voices, "model": models},
    }


@app.put("/api/settings")
async def put_settings(request: Request) -> dict:
    """Persist one or more settings. Body: a partial map of writable keys.

    Changes are saved server-side and take effect immediately for every surface
    that hits this backend - the browser and the Pi satellite alike.
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected a JSON object")

    writable = set(store.writable_keys())
    # reset: a list of keys to clear back to their defaults (the persona
    # "Reset to default" buttons). Handled separately from value updates.
    reset_keys = body.pop("reset", []) if isinstance(body.get("reset"), list) else []
    updates = {k: v for k, v in body.items() if k in writable}
    unknown = (set(body) - writable) | (set(reset_keys) - writable)
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown settings: {', '.join(sorted(unknown))}")
    if not updates and not reset_keys:
        raise HTTPException(status_code=400, detail="no settings to update")

    # Validate the voice against what the TTS service actually offers, but don't
    # block a save if TTS is unreachable - we just can't verify it then.
    if "voice" in updates:
        try:
            voices = await tts.list_voices()
        except httpx.HTTPError:
            voices = []
        if voices and updates["voice"] not in voices:
            raise HTTPException(status_code=400, detail=f"unknown voice: {updates['voice']}")

    # Likewise validate the model against what's installed, when reachable.
    if "model" in updates:
        models = await llm.list_models()
        if models and updates["model"] not in models:
            raise HTTPException(status_code=400, detail=f"unknown model: {updates['model']}")

    for key in reset_keys:
        store.reset(key)
    for key, value in updates.items():
        store.set(key, str(value))
    return {"settings": store.all()}


@app.post("/api/notifications/test")
async def test_notification() -> dict:
    """Publish a test notification to the configured broker/topic so the user can
    verify the MQTT → Home Assistant → phone pipe end to end. Bypasses the enable
    /category gate (so it works during setup) but still needs the address + topic
    set. Backs the 'Send test' button in Settings ▸ Notifications."""
    from .services import notify
    if not await notify.send_test():
        raise HTTPException(
            status_code=400,
            detail="Test failed — check the broker address/topic and the activity log.",
        )
    return {"sent": True}


@app.get("/api/ha/devices")
async def ha_devices(refresh: bool = False) -> dict:
    """The curated (labeled) devices the assistant can control - for the Settings
    panel's Home Assistant section. `?refresh=1` forces a re-fetch from HA, used
    right after labeling a device so it shows up without waiting for the cache."""
    from .services import home_assistant as ha
    if not settings.ha_enabled:
        return {"enabled": False, "label": settings.ha_label, "devices": []}
    if refresh:
        await ha.refresh_devices()
    else:
        await ha.ensure_fresh()
    return {"enabled": True, "label": settings.ha_label, "devices": ha.devices()}


# Cameras (Frigate, #68)
# The PWA addresses a camera only by id under /api/cameras/<id>/. These routes
# resolve the id against Frigate's catalog (refusing anything not on it, a small
# SSRF guard) and proxy same-origin, so the browser never talks to Frigate.
# go2rtc hands us browser-playable HLS, plus a JPEG snapshot as an instant poster.

def _camera_or_404(camera_id: str) -> dict:
    from .services import frigate
    cam = frigate.by_id(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"unknown camera: {camera_id}")
    return cam


async def _proxy_frigate(path: str, params: dict | None = None) -> StreamingResponse:
    """Stream an upstream Frigate response back to the browser, passing its status
    and content-type through unchanged (used for the snapshot + HLS proxy)."""
    from .services import frigate
    client, resp = await frigate.open_upstream(path, params)

    async def body():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        body(), status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/cameras")
async def list_cameras() -> dict:
    """Available cameras (id + friendly name) for the PWA to resolve a feed against.
    Empty when Frigate isn't configured."""
    from .services import frigate
    if not settings.frigate_enabled:
        return {"enabled": False, "cameras": []}
    await frigate.ensure_fresh()
    return {"enabled": True,
            "cameras": [{"id": c["id"], "name": c["name"]} for c in frigate.cameras()]}


@app.get("/api/cameras/{camera_id}/snapshot.jpg")
async def camera_snapshot(camera_id: str) -> StreamingResponse:
    """A latest-frame JPEG - the modal's poster while HLS buffers."""
    from .services import frigate
    await frigate.ensure_fresh()
    cam = _camera_or_404(camera_id)
    return await _proxy_frigate(frigate.snapshot_path(cam["id"]))


@app.get("/api/cameras/{camera_id}/index.m3u8")
async def camera_hls_master(camera_id: str) -> StreamingResponse:
    """The HLS master playlist for a camera's live feed (go2rtc, proxied)."""
    from .services import frigate
    await frigate.ensure_fresh()
    cam = _camera_or_404(camera_id)
    path, params = frigate.hls_master_path(cam["stream"])
    return await _proxy_frigate(path, params)


@app.get("/api/cameras/{camera_id}/hls/{rest:path}")
async def camera_hls_child(camera_id: str, rest: str, request: Request) -> StreamingResponse:
    """HLS sub-playlists + segments. The master references these relatively, so the
    browser resolves them under this camera's base path; we map them straight back
    onto go2rtc's HLS tree (no URL rewriting)."""
    from .services import frigate
    await frigate.ensure_fresh()
    _camera_or_404(camera_id)
    return await _proxy_frigate(frigate.hls_sub_path(rest), dict(request.query_params))


@app.get("/api/conversations")
async def list_conversations(days: int | None = None) -> dict:
    """Recent conversations (rolling window), most recently used first.

    Spans every surface - a chat started on the satellite appears here too.
    """
    return {"conversations": history.list_conversations(days), "days": days or settings.history_days}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict:
    """One conversation with its full message list, for loading/continuing it."""
    conv = history.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict:
    """Delete a conversation and its messages."""
    if not history.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"deleted": conversation_id}


@app.get("/api/memories")
async def list_memories() -> dict:
    """Active (non-expired) memories for the Settings → Memory tab, long→short."""
    memory.prune()
    return {"memories": memory.list_active(), "now": memory._now()}


@app.post("/api/memories")
async def create_memory(request: Request) -> dict:
    """Add a hand-curated memory. Body: {"text": str, "tier": short|mid|long}."""
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    tier = body.get("tier") if body.get("tier") in memory.TIERS else "long"
    m = memory.add(text, tier=tier, source="web")
    log.info("memory added [%s] %r", tier, _snippet(text))
    return {"memory": m}


@app.put("/api/memories/{memory_id}")
async def update_memory(memory_id: str, request: Request) -> dict:
    """Edit a memory's text and/or tier (re-computes its expiry from the tier)."""
    body = await request.json()
    tier = body.get("tier") if body.get("tier") in memory.TIERS else None
    m = memory.update(memory_id, text=body.get("text"), tier=tier)
    if m is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"memory": m}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str) -> dict:
    """Forget a memory (the Memory tab's ✕)."""
    if not memory.delete(memory_id):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": memory_id}


@app.get("/api/log")
async def get_log(limit: int = 500, since: int = 0, before: int = 0,
                  level: str | None = None, hide_health: bool = False) -> dict:
    """Recent activity log for the Settings-screen viewer (#21).

    Newest first. ``since`` returns only rows past that id (incremental polling for
    new activity); ``before`` returns only rows older than that id (paging back
    through the 48h history); ``level`` filters to that severity and above;
    ``hide_health`` drops the routine health/polling chatter.
    """
    entries = logstore.recent(limit=limit, since=since, before=before,
                              level=level, hide_noise=hide_health)
    return {"log": entries, "retention_hours": settings.log_retention_hours}


@app.delete("/api/log")
async def clear_log() -> dict:
    """Empty the activity log (the viewer's Clear button)."""
    logstore.clear()
    log.info("activity log cleared from the web UI")
    return {"cleared": True}


# Shop manuals (RAG, #63)

_MANUAL_MAX_BYTES = 30 * 1024 * 1024  # 30 MB - generous for a PDF manual


@app.get("/api/manuals")
async def list_manuals() -> dict:
    """Uploaded manuals for the Settings ▸ Manuals tab (newest first)."""
    return {"manuals": manuals.list_all(), "enabled": settings.manuals_enabled}


class _ManualRejected(Exception):
    """A single file failed validation - carries a human-readable reason."""


async def _read_manual_upload(file: UploadFile) -> bytes:
    """Validate one uploaded PDF and return its bytes, or raise ``_ManualRejected``
    with a reason. Shared by the single and batch upload paths."""
    name = file.filename or "manual.pdf"
    if not name.lower().endswith(".pdf"):
        raise _ManualRejected("only PDF files are supported")
    data = await file.read()
    if not data:
        raise _ManualRejected("empty file")
    if len(data) > _MANUAL_MAX_BYTES:
        raise _ManualRejected(f"too large (max {_MANUAL_MAX_BYTES // (1024 * 1024)} MB)")
    return data


@app.post("/api/manuals")
async def upload_manual(file: UploadFile) -> dict:
    """Stage a single PDF manual for indexing. Returns the ``pending`` record
    immediately; the background worker extracts, embeds, and indexes it into Qdrant
    so ``search_manuals`` can find it. Poll ``GET /api/manuals`` for status."""
    if not settings.manuals_enabled:
        raise HTTPException(status_code=503, detail="Manuals aren't configured — set NOVA_QDRANT_URL.")
    try:
        data = await _read_manual_upload(file)
    except _ManualRejected as exc:
        # Preserve the old status semantics: bad type → 400, oversize → 413.
        reason = str(exc)
        code = 413 if "too large" in reason else 400
        raise HTTPException(status_code=code, detail=reason[:1].upper() + reason[1:] + ".")
    record = await manuals.stage(file.filename or "manual.pdf", data)
    return {"manual": record}


@app.post("/api/manuals/batch")
async def upload_manuals(files: list[UploadFile] = File(...)) -> dict:
    """Bulk-import many PDF manuals at once. Each valid file is staged and queued for
    background indexing (returned in ``staged`` as a ``pending`` record); files that
    fail validation are reported in ``errors`` without aborting the rest. Watch the
    library (``GET /api/manuals``) for each one to move to ``ready``."""
    if not settings.manuals_enabled:
        raise HTTPException(status_code=503, detail="Manuals aren't configured — set NOVA_QDRANT_URL.")
    staged: list[dict] = []
    errors: list[dict] = []
    for file in files:
        name = file.filename or "manual.pdf"
        try:
            data = await _read_manual_upload(file)
        except _ManualRejected as exc:
            errors.append({"filename": name, "error": str(exc)})
            continue
        staged.append(await manuals.stage(name, data))
    log.info("manuals: bulk import staged %d, rejected %d", len(staged), len(errors))
    return {"staged": staged, "errors": errors}


@app.get("/api/manuals/{manual_id}/file")
async def get_manual_file(manual_id: str) -> FileResponse:
    """Serve a manual's stored PDF (the original upload, kept on the ``/data``
    volume). ``inline`` so a browser/PDF viewer can render it in place."""
    record = manuals.get(manual_id)
    if record is None:
        raise HTTPException(status_code=404, detail="manual not found")
    path = manuals.file_path(manual_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="PDF is no longer stored")
    return FileResponse(
        path, media_type="application/pdf",
        filename=record["filename"], content_disposition_type="inline",
    )


@app.delete("/api/manuals/{manual_id}")
async def delete_manual(manual_id: str) -> dict:
    """Remove a manual: its Qdrant vectors, its stored file, and its registry row."""
    if not await manuals.delete(manual_id):
        raise HTTPException(status_code=404, detail="manual not found")
    log.info("manual deleted: %s", manual_id)
    return {"deleted": manual_id}


@app.get("/api/timers")
async def get_timers() -> dict:
    """Active timers & reminders for the tasks panel, plus any alarms currently
    ringing (fired and not yet dismissed). ``now`` lets the client correct for
    clock skew when rendering countdowns."""
    return {
        "timers": [format_timer(t) for t in timers.list_active()],
        "ringing": [format_timer(t) for t in timers.ringing()],
        "now": timers._now(),
    }


@app.get("/api/alarms/ringing")
async def alarms_ringing() -> dict:
    """Alarms currently ringing - the satellite polls this to loop the chime."""
    return {"ringing": [format_timer(t) for t in timers.ringing()], "now": timers._now()}


@app.delete("/api/alarms/{alarm_id}")
async def dismiss_alarm(alarm_id: str) -> dict:
    """Dismiss (acknowledge) a ringing alarm - the panel's Dismiss button."""
    if not timers.acknowledge(alarm_id):
        raise HTTPException(status_code=404, detail="alarm not ringing")
    return {"dismissed": alarm_id}


@app.get("/api/timers/fired")
async def timers_fired(since: float = 0) -> dict:
    """Timers that fired after ``since`` - the satellite's announce feed. It
    advances ``since`` to the returned ``now`` so each firing is announced once."""
    return {"fired": timers.fired_since(since), "now": timers._now()}


@app.delete("/api/timers/{timer_id}")
async def cancel_timer(timer_id: str) -> dict:
    """Cancel an active timer/reminder (the panel's ✕ and voice both use this)."""
    if not timers.cancel(timer_id):
        raise HTTPException(status_code=404, detail="timer not found or not active")
    return {"cancelled": timer_id}


@app.get("/api/satellites")
async def get_satellites() -> dict:
    """Known satellites with online status + desired state, for the panel."""
    return {"satellites": satellites.list_all()}


@app.post("/api/satellites/{satellite_id}/heartbeat")
async def satellite_heartbeat(satellite_id: str, request: Request) -> dict:
    """A satellite checks in, reporting its physical-button (hardware) mute state;
    we record it and return the desired software mute + playback volume for it to
    apply locally."""
    body = await request.json()
    row = satellites.heartbeat(
        satellite_id,
        body.get("name") or satellite_id,
        hw_muted=body.get("hw_muted"),
    )
    return {"muted": bool(row["muted"]), "volume": row["volume"]}


@app.put("/api/satellites/{satellite_id}")
async def update_satellite(satellite_id: str, request: Request) -> dict:
    """Set a satellite's desired mic/volume state and/or display name (web panel)."""
    body = await request.json()
    row = satellites.set_state(satellite_id, muted=body.get("muted"),
                               volume=body.get("volume"), name=body.get("name"))
    if row is None:
        raise HTTPException(status_code=404, detail="satellite not found")
    return row


@app.delete("/api/satellites/{satellite_id}")
async def delete_satellite(satellite_id: str) -> dict:
    """Forget a satellite (e.g. a decommissioned device)."""
    if not satellites.forget(satellite_id):
        raise HTTPException(status_code=404, detail="satellite not found")
    return {"deleted": satellite_id}


# Link dashboard (Homepage replacement)
@app.get("/api/dashboard")
async def get_dashboard() -> dict:
    """The current dashboard document (the user's saved copy, else the seed)."""
    return dashboard.store.get()


@app.put("/api/dashboard")
async def put_dashboard(request: Request) -> dict:
    """Replace the whole dashboard document (edited in the UI). Validated/sanitized
    server-side, then persisted; returns the stored form so the client re-syncs."""
    body = await request.json()
    return dashboard.store.save(body)


@app.post("/api/dashboard/reset")
async def reset_dashboard() -> dict:
    """Drop the user's edits and fall back to the packaged seed."""
    return dashboard.store.reset()


# Live homelab metrics (constellation HUD)
@app.get("/api/metrics")
async def api_metrics() -> dict:
    """Per-host system vitals (Beszel) + Frigate, for
    the constellation HUD. Cached server-side (``metrics_ttl``); best-effort - a down
    source just drops out. Empty/``enabled:false`` when no metric source is set."""
    if not settings.metrics_enabled:
        return {"enabled": False, "hosts": [], "frigate": None}
    data = await metrics_svc.get_metrics()
    return {"enabled": True, **data}


@app.post("/api/stt")
async def api_stt(audio: UploadFile) -> dict:
    """Transcribe an uploaded audio clip (WebM/Opus from MediaRecorder)."""
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        result = await stt.transcribe(data, content_type=audio.content_type or "audio/webm")
    except httpx.HTTPError as exc:
        log.error("STT upstream error (%d bytes): %s", len(data), exc)
        raise HTTPException(status_code=502, detail=f"stt upstream error: {exc}") from exc
    text = result.get("text", "")
    log.info("STT transcribed %d bytes → %r", len(data), _snippet(text))
    return {"text": text, "meta": result}


@app.post("/api/chat")
async def api_chat(request: Request) -> StreamingResponse:
    """Stream an assistant reply as Server-Sent Events.

    Body: {"message": str, "conversation_id": str|null, "source": str, "mode": str}.
    The server owns the session (#44): it rehydrates the thread from stored history
    and assembles the model context - the client sends only the new user message.
    Events: `data: {"delta": "..."}` per token, then `data: {"done": true}`.
    """
    body = await request.json()
    # Server-owned sessions (#44): the client sends only the new user message plus
    # its conversation id; the server rehydrates the thread and assembles context.
    # History is the source of truth, so a thread continues across surfaces.
    new_user = (body.get("message") or "").strip()
    if not new_user:
        raise HTTPException(status_code=400, detail="message required")
    # "voice" (default) keeps replies short and speakable; "text" allows richer,
    # markdown-friendly answers for the browser text interface.
    mode = "text" if body.get("mode") == "text" else "voice"
    # Which chat this turn belongs to (a new one is minted on persist if absent)
    # and which surface it came from, for the history log.
    conversation_id = body.get("conversation_id") or None
    source = "satellite" if body.get("source") == "satellite" else "web"

    # Rehydrate the thread from the shared store and append the new turn. The full
    # stored thread is replayed (conversations are short at single-user scale); if
    # a pathological thread ever strains the model's context window, bound it here.
    messages = history.thread_for(conversation_id) + [{"role": "user", "content": new_user}]

    # Root span for the turn. Started explicitly rather than with a `with` block:
    # the reply streams from an async generator below, and OTel's implicit
    # current-span context does not survive a `yield`.
    turn_span = tracing.start(
        "chat.turn",
        kind=tracing.AGENT,
        **{tracing.INPUT_VALUE: new_user, tracing.SESSION_ID: conversation_id or ""},
    )

    recent_user = [m["content"] for m in messages if m["role"] == "user"][-6:]
    # The router is the turn's first decision and the biggest source of wrong
    # answers. It hands the model only the routed skills' tools and prompt
    # fragments, and takes recent user messages so a pronoun follow-up inherits
    # the skill it continues. No model call, so CHAIN is the closest span kind.
    with tracing.span("route", kind=tracing.CHAIN, parent=turn_span.raw) as route_span:
        route_span.set_input(recent_user)
        skill_names = registry.route(recent_user)
        tools = registry.tools_for(skill_names)
        system_note = registry.context_for(skill_names)
        route_span.set_output(skill_names or [])
        route_span.set(**{tracing.METADATA: {
            "skills": skill_names,
            "tools_offered": [s["function"]["name"] for s in (tools or [])],
        }})
    log.info("chat turn [%s/%s] %r → skills=%s", source, mode, _snippet(new_user), skill_names or "none")

    # Auto-escalation gate. When on, hard turns run on Claude with the same routed
    # tools: upfront (the request clearly wants the big brain, so skip the wasted
    # qwen3 pass) or reactive (qwen3 tried and struggled). Off means nothing
    # leaves the LAN.
    auto_escalate = settings.claude_enabled and store.get("auto_escalate") == "1"
    # The escalation decision gets its own span. Not a model call and not a tool
    # call, but the most consequential choice in the turn (cost, latency, whether
    # data leaves the LAN), so it has to be queryable. CHAIN is the closest kind.
    with tracing.span("escalation.decide", kind=tracing.CHAIN, parent=turn_span.raw) as esc:
        esc.set_input(new_user)
        upfront = auto_escalate and escalation.should_escalate_upfront(new_user)
        esc.set_output("claude-deep" if upfront else "local-first")
        esc.set(**{tracing.METADATA: {
            "phase": "upfront",
            "auto_escalate": auto_escalate,
            "claude_enabled": settings.claude_enabled,
        }})
    turn_span.add_metadata({
        "source": source,
        "mode": mode,
        "skills": skill_names,
        "auto_escalate": auto_escalate,
        "upfront_escalation": upfront,
        "history_turns": len(messages) - 1,
    })

    async def event_stream():
        # Tag any timer/reminder the agent creates this turn with the surface it
        # came from. Set inside the generator so it's in the streaming context.
        timers.current_source.set(source)
        reply_parts: list[str] = []
        tool_calls: list[dict] = []
        errored = False

        # Capture the manuals a search_manuals call cited (id + page), so the client
        # can offer "open the PDF at that page" chips under the reply. Wraps the real
        # executor so both the local and Claude agent loops feed it, without either
        # loop having to know about manuals.
        manual_sources: list[dict] = []

        async def execute_tool(name, args):
            # One span per dispatch, parented to the turn rather than the LLM round
            # that asked for it: that round's span has already closed, and nesting
            # under a finished span renders as a child starting after its parent.
            bare = name.split(TOOL_NS_SEP, 1)[-1]
            retrieval = bare == "search_manuals"
            span = tracing.start(
                f"tool.{bare}",
                # The manuals search IS retrieval, so it gets the RETRIEVER kind
                # and document attributes - otherwise RAG quality is invisible.
                kind=tracing.RETRIEVER if retrieval else tracing.TOOL,
                parent=turn_span.raw,
                **{
                    tracing.TOOL_NAME: name,
                    tracing.TOOL_PARAMETERS: args,
                    tracing.INPUT_VALUE: args.get("query", "") if retrieval else args,
                },
            )
            try:
                result = await registry.execute(name, args)
            except Exception as exc:  # noqa: BLE001 - record, then let it surface
                span.record_error(exc)
                span.end()
                raise
            if retrieval and isinstance(result, dict):
                hits = result.get("results") or []
                span.set(**tracing.document_attributes([
                    {
                        "id": r.get("id"),
                        "content": r.get("text"),
                        "score": r.get("score"),
                        "metadata": {"manual": r.get("manual"), "page": r.get("page")},
                    }
                    for r in hits
                ]))
                span.set(**{tracing.METADATA: {"hits": len(hits)}})
                for r in hits:
                    if r.get("id"):
                        manual_sources.append(
                            {"id": r["id"], "title": r.get("manual"), "page": r.get("page")})
            span.set_output(result)
            # A provider that returns {"error": …} has failed without raising -
            # mark the span so failed tools are filterable, not just readable.
            if isinstance(result, dict) and result.get("error"):
                span.set(**{tracing.METADATA: {"tool_error": str(result["error"])[:200]}})
            span.end()
            return result

        # A run of the Claude brain (agent loop over the same tools), streamed live.
        # model picks the tier - the fast/cheap model reactively, the deep model
        # upfront.
        async def run_claude(model):
            nonlocal errored
            async for event in claude.agent_reply(
                messages, tools=tools, execute_tool=execute_tool,
                system_note=system_note, response_mode=mode, model=model,
                parent=turn_span,
            ):
                kind = event["type"]
                if kind == "delta":
                    reply_parts.append(event["text"])
                    yield _sse({"delta": event["text"]})
                elif kind == "tool":
                    bare = event["name"].split(TOOL_NS_SEP, 1)[-1]
                    log.info("tool call [claude] %s(%s)", bare, _snippet(json.dumps(event["arguments"]), 160))
                    tool_calls.append({"tool": bare, "args": event["arguments"]})
                    yield _sse({"tool": bare, "args": event["arguments"]})
                elif kind == "error":
                    errored = True
                    log.error("escalation error [%s]: %s", source, event["message"])
                    yield _sse({"error": event["message"]})

        escalated = False
        claude_model_used = None  # the exact Claude model, when a tier is used
        if upfront:
            # Straight to the DEEP brain - the request wants big-model reasoning.
            escalated = True
            claude_model_used = settings.claude_model
            log.info("upfront escalation to Claude deep (%s): %r", claude_model_used, _snippet(new_user))
            yield _sse({"escalated": True, "tier": "deep"})
            async for chunk in run_claude(claude_model_used):
                yield chunk
            reply = "".join(reply_parts).strip()
        else:
            # Local first. Buffer qwen3's text when escalation is on (so a struggle
            # can be replaced cleanly); stream live when it's off.
            async for event in llm.agent_reply(
                messages, tools=tools, execute_tool=execute_tool,
                system_note=system_note, response_mode=mode,
                parent=turn_span.raw,
            ):
                kind = event["type"]
                if kind == "delta":
                    reply_parts.append(event["text"])
                    if not auto_escalate:
                        yield _sse({"delta": event["text"]})
                elif kind == "tool":
                    bare = event["name"].split(TOOL_NS_SEP, 1)[-1]
                    log.info("tool call %s(%s)", bare, _snippet(json.dumps(event["arguments"]), 160))
                    tool_calls.append({"tool": bare, "args": event["arguments"]})
                    yield _sse({"tool": bare, "args": event["arguments"]})
                elif kind == "error":
                    errored = True
                    log.error("chat error [%s]: %s", source, event["message"])
                    yield _sse({"error": event["message"]})

            reply = "".join(reply_parts).strip()
            # The reactive half of the decision: qwen3 answered - was it good
            # enough? Traced whether or not it escalates, so the no cases are
            # countable too; a struggle detector is only assessable against the
            # turns it declined to escalate.
            struggled = False
            if auto_escalate and not errored:
                with tracing.span("escalation.decide", kind=tracing.CHAIN, parent=turn_span.raw) as esc:
                    esc.set_input(reply)
                    struggled = escalation.is_struggle(reply, tool_calls)
                    esc.set_output("claude-fast" if struggled else "keep-local")
                    esc.set(**{tracing.METADATA: {
                        "phase": "reactive",
                        "tool_calls": [t["tool"] for t in tool_calls],
                        "local_reply_chars": len(reply),
                    }})
            if struggled:
                # qwen3 fell short - discard it and retry on the FAST/cheap brain
                # (cheap enough to escalate liberally; reliable on tool chains).
                escalated = True
                claude_model_used = settings.claude_fast_model
                log.info("reactive escalation to Claude fast (%s): local reply=%r",
                         claude_model_used, _snippet(reply))
                yield _sse({"escalated": True, "tier": "fast"})
                reply_parts.clear()
                manual_sources.clear()  # discard qwen3's citations; Claude re-runs its own
                async for chunk in run_claude(claude_model_used):
                    yield chunk
                if reply_parts:
                    reply = "".join(reply_parts).strip()  # Claude's answer replaces qwen3's
            elif auto_escalate:
                # Buffered but no escalation - flush qwen3's answer now.
                if reply:
                    yield _sse({"delta": reply})

        # Camera safety net (#68). qwen3:8b sometimes says "here's the front porch
        # camera" without calling show_camera, so the modal never opens. If the turn
        # routed to cameras and the reply claims a camera is up but none fired,
        # synthesize the event. Gated on the model's own claim, so it won't fire on
        # "is the driveway recording?".
        if ("cameras" in skill_names
                and not any(t["tool"] == "show_camera" for t in tool_calls)
                and frigate_svc.looks_like_show_confirmation(reply)):
            await frigate_svc.ensure_fresh()
            # Prefer the camera the model named in its reply ("here's the Shop
            # Inside camera") over the raw request, which may be ambiguous
            # ("the shop camera"); fall back to the request if the reply doesn't
            # pin one down.
            cam = frigate_svc.resolve(reply) or frigate_svc.resolve(new_user)
            if cam:
                args = {"camera": cam["id"]}
                tool_calls.append({"tool": "show_camera", "args": args})
                log.info("camera fallback: model skipped show_camera; emitting %s", cam["id"])
                yield _sse({"tool": "show_camera", "args": args})

        # Cited manuals (dedup by doc+page, best-scored first) → clickable "open the
        # PDF at that page" chips under the reply.
        if manual_sources:
            seen, uniq = set(), []
            for s in manual_sources:
                key = (s["id"], s["page"])
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(s)
            yield _sse({"sources": uniq[:6]})

        # Tag the turn with the model that produced the final reply (the fast or deep
        # Claude tier when escalated, else the local model) - surfaced as a chip.
        model_used = claude_model_used if escalated else store.get("model")
        yield _sse({"model": model_used})

        # Persist the exchange and tell the client which conversation it landed
        # in, so it can group follow-up turns and show it in the history list.
        cid = conversation_id
        if new_user:
            try:
                cid = history.record_turn(conversation_id, source, new_user, reply,
                                          tool_calls=tool_calls, escalated=escalated,
                                          model=model_used)
            except Exception as exc:  # noqa: BLE001 - never let logging break the reply
                yield _sse({"error": f"history write failed: {exc}"})
        if cid:
            yield _sse({"conversation": cid})

        # Close out the turn span's attributes. The conversation id is only known
        # now (it is minted on first persist), which is why session.id is set at
        # the end rather than at span creation.
        turn_span.set_output(reply)
        turn_span.set(**{tracing.SESSION_ID: cid or ""})
        turn_span.add_metadata({
            "model": model_used,
            "escalated": escalated,
            "tool_calls": [t["tool"] for t in tool_calls],
            "errored": errored,
        })
        yield _sse({"done": True})

    async def traced_stream():
        """Guarantees the turn span is closed even if the client disconnects
        mid-stream - otherwise an abandoned SSE response leaks an open span."""
        try:
            async for chunk in event_stream():
                yield chunk
        finally:
            turn_span.end()

    return StreamingResponse(traced_stream(), media_type="text/event-stream")


@app.post("/api/tts")
async def api_tts(request: Request) -> StreamingResponse:
    """Synthesise speech for `text`, streaming mp3 audio back."""
    body = await request.json()
    text = _clean_for_speech((body.get("text") or "").strip())
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    # A caller may still pin a per-request voice, but with none given we use the
    # persisted setting - so the browser and the satellite speak in the voice
    # chosen in the Settings screen.
    voice = body.get("voice") or store.get("voice")
    fmt = (body.get("format") or tts.AUDIO_FORMAT).lower()

    async def audio_stream():
        try:
            async for chunk in tts.stream_speech(text, voice=voice, fmt=fmt):
                yield chunk
        except httpx.HTTPError:
            # The stream has already started (status 200 sent); we can only stop.
            return

    return StreamingResponse(audio_stream(), media_type=tts.media_type(fmt))


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _snippet(text: str, limit: int = 80) -> str:
    """A short, single-line preview of text for the activity log."""
    line = " ".join((text or "").split())
    return line[:limit] + ("…" if len(line) > limit else "")


# Strip markdown + emoji so TTS never reads out asterisks, "- [ ]" checkboxes, or
# symbol names like "heavy check mark" / "smiling face with smiling eyes". The
# model is told to avoid all of this, but confirmations - and reading notes back -
# still slip some in.
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_MARKS = re.compile(r"[*_`#>]+")
# Leading bullet / task-checkbox markers, per line ("- [ ] milk" -> "milk").
_LIST_MARK = re.compile(r"(?m)^[ \t]*[-*+][ \t]+(?:\[[ xX]\][ \t]*)?")
# Emoji, dingbats, and misc symbols (incl. ✔ U+2714, ☑, ✅, 😊, ⭐, ⏰) plus the
# variation selectors / zero-width joiners that glue multi-codepoint emoji.
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U00002b00-\U00002bff"
    "\U00002190-\U000021ff\U00002300-\U000023ff\U000025a0-\U000025ff"
    "\U0001f1e6-\U0001f1ff\U0000fe00-\U0000fe0f\U0000200d\U000020e3]+"
)


def _clean_for_speech(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _LIST_MARK.sub("", text)
    text = _MD_MARKS.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# Static PWA
# Mounted last so /api/* wins. The service worker and manifest must be served
# from the root scope, which StaticFiles handles.

if os.path.isdir(settings.frontend_dir):
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
else:  # pragma: no cover - only in a misconfigured deploy
    @app.get("/")
    async def _missing_frontend() -> FileResponse:
        raise HTTPException(status_code=500, detail=f"frontend not found at {settings.frontend_dir}")
