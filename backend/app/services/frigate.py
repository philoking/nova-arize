"""Frigate NVR client - camera catalog + a same-origin proxy for live feeds (#68).

The assistant can "pull up" a camera: ``show_camera`` (see ``plugins/cameras.py``)
matches a spoken name against the cameras configured in Frigate and the PWA opens
a modal with the live video.

Two facts shape this module:

- **Browsers can't play RTSP.** Frigate bundles **go2rtc**, which restreams each
  camera's RTSP as browser-playable **HLS** (H.264/AAC). We hand the ``<video>``
  that HLS, never the RTSP URL.
- **The browser talks only to the nova-voice backend.** So we *proxy* Frigate
  server-side (like the AI services): the frontend hits ``/api/cameras/...`` and
  this module forwards to Frigate. That keeps it same-origin (no CORS, no mixed
  content) and means the browser never learns Frigate's address.

The camera list (id, friendly name, go2rtc stream) is fetched once from Frigate's
``/api/config``, cached, and refreshed on a background loop - the same pattern as
the Home Assistant device allowlist. ``show_camera`` and the proxy routes both
resolve a requested camera against this cached set, so an unknown/garbage id is
refused rather than blindly forwarded (a small SSRF guard on the proxy).
"""

from __future__ import annotations

import re
import time

import httpx

from ..config import settings

# HTTP client

def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.frigate_url,
        timeout=15,
        verify=settings.frigate_verify_ssl,
    )


# Camera catalog (cached)
# Each entry: {"id": <frigate camera key>, "name": <friendly>, "stream": <go2rtc>}.
# Assigned atomically on refresh so readers see a consistent list without a lock.
_cameras: list[dict] = []
_fetched_at: float = 0.0

# Tokens dropped when turning a camera id into a friendly name / match haystack -
# every camera is a "camera", so the word carries no signal.
_NAME_STOPWORDS = {"camera", "cam", "main", "sub"}
# Filler words dropped from a spoken query before matching (mirrors home_assistant).
_QUERY_STOPWORDS = {"the", "a", "an", "to", "my", "on", "of", "and", "please", "show",
                    "me", "pull", "up", "see", "let", "look", "at", "view", "watch",
                    "open", "display", "camera", "cam", "feed", "live", "stream"}


def _tokens(text: str, stop: set[str]) -> set[str]:
    """Lowercase word tokens minus stopwords, with naive plural stemming."""
    out = set()
    for raw in re.split(r"[^a-z0-9]+", (text or "").lower()):
        if len(raw) < 2 or raw in stop:
            continue
        out.add(raw[:-1] if len(raw) > 3 and raw.endswith("s") else raw)
    return out


def _friendly(camera_id: str) -> str:
    """A human name for a camera id: 'front_porch_camera' -> 'Front Porch'."""
    words = [w for w in re.split(r"[^a-z0-9]+", camera_id.lower())
             if w and w not in _NAME_STOPWORDS]
    return " ".join(w.capitalize() for w in words) or camera_id


def _stream_for(camera_id: str, cfg: dict) -> str:
    """The go2rtc stream to play for a camera.

    Frigate records the live streams under ``cameras.<id>.live.streams`` as a
    ``{label: go2rtc_stream}`` map; we take the first (the main/high-quality one).
    Falls back to ``<base>_main`` - go2rtc's conventional name for the primary
    stream - when a camera doesn't declare one explicitly.
    """
    streams = ((cfg.get("live") or {}).get("streams") or {})
    if isinstance(streams, dict) and streams:
        first = next(iter(streams.values()))
        if isinstance(first, str) and first:
            return first
    base = re.sub(r"_camera$", "", camera_id)
    return f"{base}_main"


async def refresh_cameras() -> list[dict]:
    """Re-fetch the camera catalog from Frigate. On any error the previous cache
    is kept (a transient Frigate blip shouldn't empty the list)."""
    global _cameras, _fetched_at
    if not settings.frigate_enabled:
        return []
    try:
        async with _client() as client:
            resp = await client.get("/api/config")
            resp.raise_for_status()
            cfg = resp.json()
    except (httpx.HTTPError, ValueError):
        _fetched_at = time.monotonic()  # don't hammer Frigate on a persistent error
        return _cameras
    cams = cfg.get("cameras") if isinstance(cfg, dict) else None
    if isinstance(cams, dict):
        items = [
            {"id": cid, "name": _friendly(cid), "stream": _stream_for(cid, ccfg or {})}
            for cid, ccfg in cams.items()
        ]
        _cameras = sorted(items, key=lambda c: c["name"])
    _fetched_at = time.monotonic()
    return _cameras


def cameras() -> list[dict]:
    """The cached camera catalog (may be up to ``frigate_cameras_ttl`` stale)."""
    return _cameras


async def ensure_fresh() -> None:
    if not _cameras or (time.monotonic() - _fetched_at) > settings.frigate_cameras_ttl:
        await refresh_cameras()


async def refresh_loop() -> None:
    """Keep the catalog warm so a camera added in Frigate shows up without a
    restart. Started at app startup when Frigate is configured."""
    import asyncio
    while True:
        await asyncio.sleep(max(30, settings.frigate_cameras_ttl))
        try:
            await refresh_cameras()
        except Exception:  # noqa: BLE001 - long-running daemon loop
            pass


def by_id(camera_id: str) -> dict | None:
    return next((c for c in _cameras if c["id"] == camera_id), None)


def resolve(query: str) -> dict | None:
    """Best-effort map a spoken name *or* an exact id to a camera.

    Exact id wins; otherwise rank by how many of the query's words hit the
    camera's friendly name / id. Returns the best camera, or ``None`` if the
    query shares no words with any camera (so a wrong guess opens nothing rather
    than a random feed)."""
    if not query:
        return None
    exact = by_id(query.strip())
    if exact:
        return exact
    q = _tokens(query, _QUERY_STOPWORDS)
    if not q:
        return None
    best, best_score = None, 0
    for c in _cameras:
        hay = _tokens(f"{c['name']} {c['id']}", _NAME_STOPWORDS)
        score = len(q & hay)
        if score > best_score:
            best, best_score = c, score
    return best


# qwen3:8b sometimes writes "here's the front porch camera" without actually
# calling show_camera (its tool-calling is flaky), so nothing opens. When a turn
# routed to cameras and the reply claims a camera is on screen but no tool fired,
# /api/chat uses this to decide whether to synthesize the show_camera event.
_SHOW_CONFIRM = re.compile(
    r"\b(here'?s|here is|there'?s|there is|pulling up|showing|now showing|on screen|"
    r"on your screen|take a look|displaying|putting .* on)\b",
    re.IGNORECASE,
)


def looks_like_show_confirmation(text: str) -> bool:
    """True if a reply reads like 'here's the front porch camera' - the model
    saying it displayed a camera. Used to complete the action when it forgot the
    tool call. Deliberately excludes bare 'show' so negations ('can't show it')
    don't match."""
    return bool(_SHOW_CONFIRM.search(text or ""))


async def show_camera(camera: str) -> dict:
    """Tool impl: resolve the named camera so the model can confirm it aloud.

    The actual on-screen action is driven by the frontend off the streamed tool
    event; here we just validate and echo the resolved camera back to the model
    (and list the options when nothing matches, so it can ask which)."""
    await ensure_fresh()
    if not _cameras:
        return {"error": "no cameras are available."}
    match = resolve(camera or "")
    if not match:
        return {"error": f"no camera matches {camera!r}.",
                "available": [c["name"] for c in _cameras]}
    return {"ok": True, "camera": match["id"], "name": match["name"]}


# Live-feed proxy
# The frontend only ever addresses a camera by its id under /api/cameras/<id>/…;
# these helpers forward to Frigate. A streamed upstream response is handed back
# open so the route can pass its status + content-type through unchanged and
# relay the body - the caller must close it (see main.py).

async def open_upstream(path: str, params: dict | None = None):
    """Open a streamed GET against Frigate. Returns (client, response); the caller
    relays ``response`` and must ``aclose`` both when the body is exhausted."""
    client = _client()
    req = client.build_request("GET", path, params=params or None)
    resp = await client.send(req, stream=True)
    return client, resp


def snapshot_path(camera_id: str) -> str:
    """Frigate's latest-frame JPEG for a camera - a cheap poster while HLS buffers."""
    return f"/api/{camera_id}/latest.jpg"


def hls_master_path(stream: str) -> tuple[str, dict]:
    """go2rtc's HLS master playlist for a stream (proxied via Frigate)."""
    return "/api/go2rtc/api/stream.m3u8", {"src": stream}


def hls_sub_path(rest: str) -> str:
    """An HLS sub-playlist/segment path, mapped back onto go2rtc's HLS route.

    go2rtc's master playlist references its media playlist relatively as
    ``hls/playlist.m3u8?id=…`` and that references its segments relatively in turn,
    so the browser resolves them under our ``/api/cameras/<id>/hls/`` base. Our
    route captures the part after ``/hls/`` as ``rest``; go2rtc serves those same
    children under ``/api/go2rtc/api/hls/``, so we just prepend that - no URL
    rewriting of the playlists needed."""
    return f"/api/go2rtc/api/hls/{rest.lstrip('/')}"


async def health() -> bool:
    if not settings.frigate_enabled:
        return False
    try:
        async with _client() as client:
            # Short probe timeout so a slow Frigate can't stall /api/health (the
            # shared client defaults to 15s; health is polled every 20s).
            resp = await client.get("/api/version", timeout=5)
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
