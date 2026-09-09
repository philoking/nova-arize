"""Aggregate live homelab metrics for the constellation HUD.

Per-host system vitals come from Beszel (CPU/mem/disk + a load-average history for
the sparklines), plus a small Frigate NVR readout. A short in-memory TTL cache
(``metrics_ttl``) shares one fetch across every surface that polls, so the browser
+ satellite don't each hammer the sources. Every source is best-effort - a down one
just drops out of the payload rather than failing it.
"""

from __future__ import annotations

import asyncio
import time

from ..config import settings
from . import beszel, frigate

_cache: dict | None = None
_cache_at: float = 0.0


async def _frigate() -> dict | None:
    if not settings.frigate_enabled:
        return None
    try:
        await frigate.ensure_fresh()
        return {"cameras": len(frigate.cameras())}
    except Exception:  # noqa: BLE001 - never let a source break the aggregate
        return None


async def get_metrics(force: bool = False) -> dict:
    """The aggregated HUD payload, cached for ``metrics_ttl`` seconds."""
    global _cache, _cache_at
    if _cache is not None and not force and (time.monotonic() - _cache_at) < settings.metrics_ttl:
        return _cache

    sys_list, frig = await asyncio.gather(beszel.systems(), _frigate())

    # Recent load-average history per host for the HUD sparklines (best-effort,
    # keyed by Beszel record id; token is already warm from beszel.systems()).
    load = await beszel.load_history([s["id"] for s in sys_list if s.get("id")])

    hosts = []
    for s in sys_list:
        host = dict(s)
        series = load.get(host.pop("id", None))   # id is internal - off the payload
        if series:
            host["load"] = series
        hosts.append(host)
    # Keep a stable order: online first, then by name.
    hosts.sort(key=lambda h: (h.get("status") != "up", (h.get("name") or "").lower()))

    _cache = {"hosts": hosts, "frigate": frig, "updated": int(time.time())}
    _cache_at = time.monotonic()
    return _cache
