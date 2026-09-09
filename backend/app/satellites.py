"""Connected satellites - presence + remote control, backed by SQLite.

Satellites are polling clients (no inbound connection), so control is pull-based:
each satellite heartbeats here every couple of seconds with its id + name, and
reads back its *desired* state (mic muted, playback volume) to apply locally. The
web panel sets that desired state; it persists so it survives a backend restart
and applies whenever the device next checks in. A satellite is "online" if its
last heartbeat is within ``satellite_ttl`` seconds. State lives in the shared DB
(``db.py``).
"""

from __future__ import annotations

import time

from . import db
from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS satellites (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    first_seen INTEGER NOT NULL,
    last_seen  INTEGER NOT NULL,
    muted      INTEGER NOT NULL DEFAULT 0,   -- SOFTWARE mute set from the web app (assistant ignores the wake word)
    volume     INTEGER NOT NULL DEFAULT 100, -- desired hardware playback volume (0-100), applied by the device
    hw_muted   INTEGER NOT NULL DEFAULT 0,   -- device's PHYSICAL-button mute (the red ring), reported read-only
    display_name TEXT                        -- optional user-set name (web panel); overrides the device's self-reported `name`
);
"""

# Columns added after the initial release; ALTER-added on existing DBs (SQLite has
# no "ADD COLUMN IF NOT EXISTS", so we check PRAGMA table_info first). Older DBs may
# also carry now-unused mute_cmd/hw_capable columns from an earlier design - those
# are harmless and simply ignored.
_MIGRATIONS = {
    "hw_muted": "ALTER TABLE satellites ADD COLUMN hw_muted INTEGER NOT NULL DEFAULT 0",
    "display_name": "ALTER TABLE satellites ADD COLUMN display_name TEXT",
}

_ready = False


def _now() -> int:
    return int(time.time())


def init() -> None:
    global _ready
    with db.lock:
        c = db.conn()
        c.executescript(_SCHEMA)
        have = {r[1] for r in c.execute("PRAGMA table_info(satellites)").fetchall()}
        for col, ddl in _MIGRATIONS.items():
            if col not in have:
                c.execute(ddl)
        c.commit()
        _ready = True


def _db():
    if not _ready:
        init()
    return db.conn()


def _public(row, ttl: int) -> dict:
    """A satellite row as the API/UI wants it: clean types + computed online.

    `name` is the effective label - the user's `display_name` from the web panel if
    set, else the device's self-reported name; `default_name` is always the latter
    (so the UI can hint at the underlying device and offer a reset). `muted` is the
    software mute the app controls (assistant ignores the wake word); `hw_muted` is
    the device's physical-button mute (the red ring) - read-only, the host can't set
    it on this hardware."""
    device_name = row["name"]
    # `display_name` may be absent on a row read before the migration ran - tolerate it.
    override = row["display_name"] if "display_name" in row.keys() else None
    return {
        "id": row["id"],
        "name": override or device_name,
        "default_name": device_name,
        "muted": bool(row["muted"]),
        "hw_muted": bool(row["hw_muted"]),
        "volume": row["volume"],
        "last_seen": row["last_seen"],
        "online": (_now() - row["last_seen"]) <= ttl,
    }


def heartbeat(sid: str, name: str, hw_muted: bool | None = None) -> dict:
    """Record a check-in and the device's reported physical-button (hardware) mute
    state, returning the row so the caller can hand back the desired software mute
    and playback volume for the device to apply."""
    now = _now()
    with db.lock:
        c = _db()
        c.execute(
            "INSERT INTO satellites (id, name, first_seen, last_seen, muted, volume) "
            "VALUES (?, ?, ?, ?, 0, 100) "
            "ON CONFLICT(id) DO UPDATE SET last_seen=excluded.last_seen, name=excluded.name",
            (sid, name or sid, now, now),
        )
        if hw_muted is not None:
            c.execute("UPDATE satellites SET hw_muted=? WHERE id=?", (1 if hw_muted else 0, sid))
        c.commit()
        row = c.execute("SELECT * FROM satellites WHERE id=?", (sid,)).fetchone()
    return dict(row)


def get(sid: str, ttl: int | None = None) -> dict | None:
    with db.lock:
        row = _db().execute("SELECT * FROM satellites WHERE id=?", (sid,)).fetchone()
    return _public(row, settings.satellite_ttl if ttl is None else ttl) if row else None


def list_all(ttl: int | None = None) -> list[dict]:
    """Every known satellite (online first, then by name)."""
    t = settings.satellite_ttl if ttl is None else ttl
    with db.lock:
        rows = _db().execute("SELECT * FROM satellites ORDER BY name").fetchall()
    sats = [_public(r, t) for r in rows]
    sats.sort(key=lambda s: (not s["online"], s["name"].lower()))
    return sats


def set_state(sid: str, muted: bool | None = None, volume: int | None = None,
              name: str | None = None) -> dict | None:
    """Apply a control from the web panel. `muted` sets the software mute (assistant
    ignores the wake word); `volume` sets the desired hardware playback level; `name`
    sets the user-facing display name (an empty/whitespace value reverts to the
    device's self-reported name). The display name is sticky - the per-heartbeat
    `name` update never touches it. The device's physical-button mute (`hw_muted`) is
    read-only and not changed here. Returns the updated row, or None if id unknown."""
    sets, vals = [], []
    if muted is not None:
        sets.append("muted=?")
        vals.append(1 if muted else 0)
    if volume is not None:
        sets.append("volume=?")
        vals.append(max(0, min(100, int(volume))))
    if name is not None:
        # Empty ⇒ NULL ⇒ fall back to the device's self-reported name.
        sets.append("display_name=?")
        vals.append(name.strip() or None)
    if not sets:
        return get(sid)
    vals.append(sid)
    with db.lock:
        cur = _db().execute(f"UPDATE satellites SET {', '.join(sets)} WHERE id=?", vals)
        _db().commit()
        if cur.rowcount == 0:
            return None
    return get(sid)


def forget(sid: str) -> bool:
    """Remove a satellite from the registry (e.g. a decommissioned device)."""
    with db.lock:
        cur = _db().execute("DELETE FROM satellites WHERE id=?", (sid,))
        _db().commit()
        return cur.rowcount > 0
