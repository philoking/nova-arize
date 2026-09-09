"""Beszel system-metrics hub client - per-host CPU / memory / disk for the HUD.

Beszel (beszel.dev) is a PocketBase app: authenticate once (email+password → token)
and read the ``systems`` collection, where each record carries the host's live
``info`` blob (``cpu`` %, ``mp`` mem %, ``dp`` disk %, ``t`` temp, ``u`` uptime s).
The token is cached and only re-minted on expiry or a 401. Best-effort throughout:
any error yields an empty list rather than raising, so a hub blip just empties the
readout instead of breaking the page. Reached LAN-direct (no NPM), so plain http.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from ..config import settings

# PocketBase auth tokens are long-lived; cache and re-mint on expiry or a 401.
_token: str = ""
_token_at: float = 0.0
_TOKEN_TTL = 1800.0


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=settings.beszel_url, timeout=10,
                             verify=settings.beszel_verify_ssl)


async def _auth(client: httpx.AsyncClient, force: bool = False) -> str:
    global _token, _token_at
    if _token and not force and (time.monotonic() - _token_at) < _TOKEN_TTL:
        return _token
    resp = await client.post("/api/collections/users/auth-with-password",
                             json={"identity": settings.beszel_username,
                                   "password": settings.beszel_password})
    resp.raise_for_status()
    _token = resp.json().get("token", "")
    _token_at = time.monotonic()
    return _token


def _slim(rec: dict) -> dict:
    """One Beszel system record → the compact vitals the HUD shows."""
    info = rec.get("info") or {}
    return {
        "id": rec.get("id"),                    # Beszel record id - keys the load history
        "name": rec.get("name") or rec.get("id"),
        "status": rec.get("status") or "",     # "up" / "down" / "paused"
        "cpu": info.get("cpu"),                 # percent
        "mem": info.get("mp"),                  # percent
        "disk": info.get("dp"),                 # percent (primary disk)
        # Extra filesystems (RAID arrays, mounted volumes) as {device: percent},
        # e.g. {"md0": 39.4} for Alfred's NVR RAID. The frontend friendly-labels them.
        "extra": info.get("efs") or {},
        "temp": info.get("t"),                  # °C (may be absent)
        "uptime": info.get("u"),                # seconds
    }


async def systems() -> list[dict]:
    """Every Beszel system's live vitals; ``[]`` if disabled or unreachable."""
    if not settings.beszel_enabled:
        return []
    try:
        async with _client() as client:
            token = await _auth(client)
            resp = await client.get("/api/collections/systems/records",
                                    params={"perPage": 100}, headers={"Authorization": token})
            if resp.status_code == 401:  # token went stale → re-auth once
                token = await _auth(client, force=True)
                resp = await client.get("/api/collections/systems/records",
                                        params={"perPage": 100}, headers={"Authorization": token})
            resp.raise_for_status()
            items = resp.json().get("items", [])
    except (httpx.HTTPError, ValueError):
        return []
    return [_slim(r) for r in items if isinstance(r, dict)]


def _la1(stats: dict) -> float | None:
    """The 1-minute load average from a system_stats record's ``stats`` blob.
    Beszel stores ``la`` as ``[1m, 5m, 15m]`` (older builds: a bare number)."""
    la = stats.get("la")
    if isinstance(la, (list, tuple)) and la:
        la = la[0]
    return la if isinstance(la, (int, float)) else None


async def load_history(ids: list[str]) -> dict[str, list[float]]:
    """Recent 1-minute load-average series per system id (oldest→newest), for the
    HUD sparklines. Reads the finest ``1m`` history bucket. ``{}`` if disabled or
    unreachable; a per-host failure just omits that host. Rides the metrics cache,
    so it runs at most once per ``metrics_ttl``."""
    n = settings.beszel_load_points
    if not settings.beszel_enabled or n <= 0 or not ids:
        return {}

    try:
        async with _client() as client:
            token = await _auth(client)   # cached; systems() just warmed it

            async def one(sid: str) -> tuple[str, list[float]]:
                params = {"filter": f"(system='{sid}' && type='1m')",
                          "sort": "-created", "perPage": n, "fields": "stats"}
                resp = await client.get("/api/collections/system_stats/records",
                                        params=params, headers={"Authorization": token})
                resp.raise_for_status()
                items = resp.json().get("items", [])
                # Newest-first from the API → reverse to chronological for the line.
                series = [v for it in reversed(items)
                          if (v := _la1(it.get("stats") or {})) is not None]
                return sid, series

            pairs = await asyncio.gather(*(one(s) for s in ids), return_exceptions=True)
    except (httpx.HTTPError, ValueError):
        return {}
    return {sid: series for p in pairs if isinstance(p, tuple)
            for sid, series in [p] if series}
