"""Persistent memory - durable facts about the user, tiered by how long they last.

A small SQLite store (shared ``db.py``) of facts the assistant should remember
across conversations: preferences, key people/places, constraints. Each fact has
a **tier** that sets when it's forgotten (#29):

- ``long`` - permanent (name, allergies, preferences); never expires.
- ``mid`` - this week/month; expires after ``memory_mid_days`` (default 30).
- ``short`` - just today; expires at the end of the local day.

Facts are captured by the assistant (the ``remember`` tool) or hand-curated in the
Settings → Memory tab, and the active (non-expired) set is injected into every
turn's system prompt so replies use them. Expired facts are pruned so forgetting
is automatic and tiered.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta

from . import db
from .config import settings
from .timers import _tz  # shared timezone resolver (NOVA_VOICE_TIMEZONE)

TIERS = ("short", "mid", "long")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id         TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    tier       TEXT NOT NULL DEFAULT 'long',   -- 'short' | 'mid' | 'long'
    source     TEXT NOT NULL DEFAULT 'web',     -- 'assistant' (auto) | 'web' (curated)
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    expires_at INTEGER                          -- NULL = never (long-term)
);
CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at);
"""

# Order tiers long→mid→short in listings and prompt injection.
_TIER_RANK = {"long": 0, "mid": 1, "short": 2}

_ready = False


def _now() -> int:
    return int(time.time())


def init() -> None:
    global _ready
    with db.lock:
        db.conn().executescript(_SCHEMA)
        db.conn().commit()
        _ready = True


def _db():
    if not _ready:
        init()
    return db.conn()


def _norm_tier(tier: str | None) -> str:
    return tier if tier in TIERS else "long"


def _expires_for(tier: str, now: int) -> int | None:
    """When a fact of this tier should be forgotten (epoch), or None for long."""
    if tier == "long":
        return None
    if tier == "mid":
        return now + settings.memory_mid_days * 86400
    # short: end of the current local day.
    local = datetime.fromtimestamp(now, _tz())
    end = local.replace(hour=23, minute=59, second=59, microsecond=0)
    return int(end.timestamp())


def add(text: str, tier: str = "long", source: str = "web") -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty memory text")
    tier = _norm_tier(tier)
    now = _now()
    # De-dup: if the same fact is already remembered (case-insensitive), keep the
    # existing one rather than piling up duplicates (the model sometimes re-saves).
    for existing in list_active():
        if existing["text"].lower() == text.lower():
            return existing
    mid = uuid.uuid4().hex
    with db.lock:
        _db().execute(
            "INSERT INTO memories (id, text, tier, source, created_at, updated_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mid, text, tier, source, now, now, _expires_for(tier, now)),
        )
        _db().commit()
    return get(mid)


def get(mid: str) -> dict | None:
    with db.lock:
        row = _db().execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
    return dict(row) if row else None


def list_active(limit: int | None = None) -> list[dict]:
    """Non-expired facts, long→mid→short then most-recent first."""
    now = _now()
    with db.lock:
        rows = _db().execute(
            "SELECT * FROM memories WHERE expires_at IS NULL OR expires_at > ? "
            "ORDER BY created_at DESC", (now,)).fetchall()
    items = sorted((dict(r) for r in rows), key=lambda m: (_TIER_RANK.get(m["tier"], 9),))
    return items[:limit] if limit else items


def update(mid: str, text: str | None = None, tier: str | None = None) -> dict | None:
    row = get(mid)
    if row is None:
        return None
    now = _now()
    new_text = (text.strip() if isinstance(text, str) else None) or row["text"]
    new_tier = _norm_tier(tier) if tier is not None else row["tier"]
    # Recompute expiry from the (possibly new) tier, relative to now.
    with db.lock:
        _db().execute(
            "UPDATE memories SET text=?, tier=?, updated_at=?, expires_at=? WHERE id=?",
            (new_text, new_tier, now, _expires_for(new_tier, now), mid),
        )
        _db().commit()
    return get(mid)


def delete(mid: str) -> bool:
    with db.lock:
        cur = _db().execute("DELETE FROM memories WHERE id = ?", (mid,))
        _db().commit()
        return cur.rowcount > 0


def forget_matching(query: str) -> list[dict]:
    """Delete active memories whose text contains ``query`` (case-insensitive).
    Returns the deleted rows - the assistant's ``forget`` tool uses this."""
    needle = (query or "").strip().lower()
    if not needle:
        return []
    removed = []
    with db.lock:
        for m in list_active():
            if needle in m["text"].lower():
                _db().execute("DELETE FROM memories WHERE id = ?", (m["id"],))
                removed.append(m)
        _db().commit()
    return removed


def prune() -> int:
    """Drop expired facts. Cheap; run at startup and after writes."""
    now = _now()
    with db.lock:
        cur = _db().execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
        _db().commit()
        return cur.rowcount
