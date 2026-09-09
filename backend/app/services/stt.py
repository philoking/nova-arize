"""Speech-to-text client - talks to the ceres-stt (faster-whisper) service.

The service accepts the raw WebM/Opus bytes a browser's MediaRecorder produces
on a single multipart field named ``audio`` and returns a JSON body with at
least ``text`` plus a detected language, duration, and heuristic confidence.
"""

from __future__ import annotations

import httpx

from ..config import settings


async def transcribe(audio: bytes, content_type: str = "audio/webm") -> dict:
    """Transcribe an audio clip. Returns the upstream JSON verbatim.

    Raises httpx.HTTPStatusError on a non-2xx response so the route can map it
    to a clean error for the client.
    """
    files = {"audio": ("clip.webm", audio, content_type)}
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.post(f"{settings.stt_url}/transcribe", files=files)
        resp.raise_for_status()
        data = resp.json()

    # The service returns the transcript under `transcript`; expose it as `text`
    # (what the rest of the app expects) while passing the rest through untouched.
    if "text" not in data:
        data["text"] = data.get("transcript", "")
    return data


async def health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.stt_url}/healthz")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
