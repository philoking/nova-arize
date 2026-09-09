"""Text-to-speech client - talks to the Kokoro FastAPI service.

Kokoro exposes an OpenAI-compatible ``/v1/audio/speech`` endpoint. We stream the
audio bytes straight back to the browser so playback can begin before the whole
clip is synthesised.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from ..config import settings

# mp3 is the most broadly playable format across browsers and the <audio> tag.
# wav / pcm are offered for clients that play with no decoder: the Pi satellite
# streams raw `pcm` straight into `aplay` as it renders (lowest latency, no header).
AUDIO_FORMAT = "mp3"
MEDIA_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm"}
MEDIA_TYPE = MEDIA_TYPES[AUDIO_FORMAT]


def media_type(fmt: str) -> str:
    return MEDIA_TYPES.get(fmt, MEDIA_TYPES[AUDIO_FORMAT])


async def stream_speech(text: str, voice: str | None = None, fmt: str = AUDIO_FORMAT) -> AsyncIterator[bytes]:
    """Yield synthesised audio bytes for `text` in the requested `voice`/format."""
    if fmt not in MEDIA_TYPES:
        fmt = AUDIO_FORMAT
    payload = {
        "model": "kokoro",
        "input": text,
        "voice": voice or settings.voice,
        "response_format": fmt,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        async with client.stream("POST", f"{settings.tts_url}/v1/audio/speech", json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk


async def list_voices() -> list[str]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{settings.tts_url}/v1/audio/voices")
        resp.raise_for_status()
        data = resp.json()
    return [v["id"] for v in data.get("voices", [])]


async def health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.tts_url}/health")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
