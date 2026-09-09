"""Timers & reminders - SQLite-backed, shared by the web UI and the satellite.

Users start countdown timers or time-based reminders by voice (satellite) or text
(web); both surfaces persist here and both can list/cancel. A tiny async scheduler
marks each one ``fired`` at its time. Delivery is by polling: the web panel counts
down and chimes, and the satellite polls the ``fired_since`` feed and announces
aloud - so a reminder goes off even with no browser open. State lives in the
shared DB (``db.py``).
"""

from __future__ import annotations

import asyncio
import calendar
import contextvars
import time
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import db
from .config import settings

_DOW = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

# The surface a chat turn came from, set per-request in /api/chat so tool-created
# timers are tagged web vs satellite. A ContextVar is async-safe: each request's
# streaming generator runs in its own context.
current_source: contextvars.ContextVar[str] = contextvars.ContextVar("timer_source", default="web")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS timers (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,                   -- 'timer' | 'reminder'
    label      TEXT NOT NULL,
    fire_at    INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    source     TEXT NOT NULL,                   -- 'web' | 'satellite'
    status     TEXT NOT NULL DEFAULT 'active',  -- 'active' | 'fired' | 'cancelled'
    fired_at   INTEGER,
    recurrence TEXT,                            -- NULL = one-shot; else e.g. 'daily', 'weekly:mon'
    is_alarm   INTEGER NOT NULL DEFAULT 0,      -- 1 = rings (loops) until acknowledged
    acknowledged_at INTEGER                     -- when a ringing alarm was dismissed
);
CREATE INDEX IF NOT EXISTS idx_timers_status_fire ON timers(status, fire_at);
CREATE INDEX IF NOT EXISTS idx_timers_fired ON timers(status, fired_at);
"""

_ready = False


def _now() -> int:
    return int(time.time())


def _tz():
    """The configured reminder timezone, else the server's local zone."""
    if settings.timezone:
        try:
            return ZoneInfo(settings.timezone)
        except Exception:  # noqa: BLE001 - a bad tz name falls back to local
            pass
    return datetime.now().astimezone().tzinfo


def _recur_matches(recurrence: str, weekday: int) -> bool:
    if recurrence == "daily":
        return True
    if recurrence == "weekdays":
        return weekday < 5
    if recurrence == "weekends":
        return weekday >= 5
    if recurrence.startswith("weekly:"):
        return _DOW.get(recurrence.split(":", 1)[1]) == weekday
    return False


def next_occurrence(recurrence: str | None, prev_epoch: int) -> int | None:
    """The next fire time (epoch) after ``prev_epoch`` for a recurrence, keeping the
    original local time-of-day. ``None`` for a one-shot or unrecognised recurrence.

    Supports ``daily``, ``weekdays``, ``weekends``, ``weekly:<dow>`` and
    ``monthly:<day>`` (day clamped to the month's length).
    """
    if not recurrence:
        return None
    tz = _tz()
    prev = datetime.fromtimestamp(prev_epoch, tz)
    tod = prev.time()
    if recurrence.startswith("monthly:"):
        try:
            day = int(recurrence.split(":", 1)[1])
        except ValueError:
            return None
        year, month = prev.year, prev.month + 1
        if month > 12:
            month, year = 1, year + 1
        day = min(day, calendar.monthrange(year, month)[1])
        nxt = datetime.combine(datetime(year, month, day).date(), tod, tzinfo=tz)
        return int(nxt.timestamp())
    if recurrence not in ("daily", "weekdays", "weekends") and not recurrence.startswith("weekly:"):
        return None
    d = prev.date()
    for _ in range(400):  # bounded scan - always resolves well within a year
        d = d + timedelta(days=1)
        if _recur_matches(recurrence, d.weekday()):
            return int(datetime.combine(d, tod, tzinfo=tz).timestamp())
    return None


def init() -> None:
    global _ready
    with db.lock:
        c = db.conn()
        c.executescript(_SCHEMA)
        # Migrate a table that predates newer columns.
        cols = {r[1] for r in c.execute("PRAGMA table_info(timers)").fetchall()}
        if "recurrence" not in cols:
            c.execute("ALTER TABLE timers ADD COLUMN recurrence TEXT")
        if "is_alarm" not in cols:
            c.execute("ALTER TABLE timers ADD COLUMN is_alarm INTEGER NOT NULL DEFAULT 0")
        if "acknowledged_at" not in cols:
            c.execute("ALTER TABLE timers ADD COLUMN acknowledged_at INTEGER")
        c.commit()
        _ready = True


def _db():
    if not _ready:
        init()
    return db.conn()


def create(kind: str, label: str, fire_at: int, source: str,
           recurrence: str | None = None, is_alarm: bool = False) -> dict:
    kind = "reminder" if kind == "reminder" else "timer"
    tid = uuid.uuid4().hex
    now = _now()
    with db.lock:
        _db().execute(
            "INSERT INTO timers (id, kind, label, fire_at, created_at, source, status, recurrence, is_alarm) "
            "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
            (tid, kind, (label or "").strip() or kind, int(fire_at), now, source,
             recurrence or None, 1 if is_alarm else 0),
        )
        _db().commit()
    return get(tid)


def get(tid: str) -> dict | None:
    with db.lock:
        row = _db().execute("SELECT * FROM timers WHERE id = ?", (tid,)).fetchone()
    return dict(row) if row else None


def list_active() -> list[dict]:
    """Active timers/reminders, soonest first - what the tasks panel shows."""
    with db.lock:
        rows = _db().execute(
            "SELECT * FROM timers WHERE status = 'active' ORDER BY fire_at").fetchall()
    return [dict(r) for r in rows]


def cancel(tid: str) -> bool:
    with db.lock:
        cur = _db().execute(
            "UPDATE timers SET status = 'cancelled' WHERE id = ? AND status = 'active'", (tid,))
        _db().commit()
        return cur.rowcount > 0


def cancel_by_label(label: str) -> dict | None:
    """Cancel the soonest active timer whose label matches (case-insensitive
    substring). Returns the cancelled row, or None if nothing matched."""
    needle = (label or "").strip().lower()
    if not needle:
        return None
    with db.lock:
        rows = _db().execute("SELECT * FROM timers WHERE status='active' ORDER BY fire_at").fetchall()
        for r in rows:
            if needle in r["label"].lower():
                _db().execute("UPDATE timers SET status='cancelled' WHERE id=?", (r["id"],))
                _db().commit()
                return dict(r)
    return None


def due(now: int) -> list[dict]:
    """Active timers whose time has arrived - the scheduler marks these fired."""
    with db.lock:
        rows = _db().execute(
            "SELECT * FROM timers WHERE status='active' AND fire_at <= ? ORDER BY fire_at", (now,)).fetchall()
    return [dict(r) for r in rows]


def mark_fired(tid: str) -> None:
    with db.lock:
        _db().execute(
            "UPDATE timers SET status='fired', fired_at=? WHERE id=? AND status='active'", (_now(), tid))
        _db().commit()


def fired_since(since: float) -> list[dict]:
    """One-shot-announce feed: timers/reminders that fired after ``since`` (epoch).
    Alarms are excluded - they loop via the ``ringing`` feed until dismissed."""
    with db.lock:
        rows = _db().execute(
            "SELECT * FROM timers WHERE status='fired' AND is_alarm=0 AND fired_at > ? "
            "ORDER BY fired_at", (int(since),)).fetchall()
    return [dict(r) for r in rows]


def ringing() -> list[dict]:
    """Alarms that have fired and not yet been acknowledged - they keep ringing on
    the web panel and the satellite until dismissed."""
    with db.lock:
        rows = _db().execute(
            "SELECT * FROM timers WHERE is_alarm=1 AND status='fired' AND acknowledged_at IS NULL "
            "ORDER BY fired_at").fetchall()
    return [dict(r) for r in rows]


def acknowledge(tid: str) -> bool:
    """Dismiss one ringing alarm. True if it was ringing."""
    with db.lock:
        cur = _db().execute(
            "UPDATE timers SET acknowledged_at=? WHERE id=? AND is_alarm=1 AND status='fired' "
            "AND acknowledged_at IS NULL", (_now(), tid))
        _db().commit()
        return cur.rowcount > 0


def acknowledge_ringing(label: str | None = None) -> list[dict]:
    """Dismiss ringing alarms - all of them, or those whose label matches (case-
    insensitive substring). Returns the alarms that were dismissed."""
    needle = (label or "").strip().lower()
    dismissed = []
    with db.lock:
        for r in ringing():
            if needle and needle not in r["label"].lower():
                continue
            _db().execute("UPDATE timers SET acknowledged_at=? WHERE id=?", (_now(), r["id"]))
            dismissed.append(dict(r))
        _db().commit()
    return dismissed


def prune(days: int | None = None) -> int:
    """Drop fired/cancelled timers older than the window. Active ones are kept
    regardless of age (a reminder set for next week must survive)."""
    window = settings.history_days if days is None else days
    cutoff = _now() - window * 86400
    with db.lock:
        cur = _db().execute(
            "DELETE FROM timers WHERE status IN ('fired','cancelled') AND created_at < ?", (cutoff,))
        _db().commit()
        return cur.rowcount


def rearm(t: dict) -> dict | None:
    """After a recurring reminder fires, schedule its next occurrence as a fresh
    active row (so the satellite fired-feed and the web chime still fire once per
    occurrence). Skips past any missed occurrences. Returns the new row, or None
    for a one-shot. Only one active row exists per series, so cancelling it ends
    the whole series."""
    recurrence = t.get("recurrence")
    if not recurrence:
        return None
    nxt = next_occurrence(recurrence, t["fire_at"])
    while nxt is not None and nxt <= _now():  # catch up if the server was down
        nxt = next_occurrence(recurrence, nxt)
    if nxt is None:
        return None
    return create(t["kind"], t["label"], nxt, t["source"],
                  recurrence=recurrence, is_alarm=bool(t.get("is_alarm")))


async def run_scheduler(on_fire=None, interval: float = 1.0) -> None:
    """Mark due timers ``fired`` once a second. Delivery is by polling (the web
    panel and the satellite fired-feed), so this just keeps server state correct
    and prompt. A recurring reminder re-arms its next occurrence on firing. A bad
    row must never kill the loop."""
    while True:
        try:
            for t in due(_now()):
                mark_fired(t["id"])
                rearm(t)
                if on_fire:
                    on_fire(t)
        except Exception:  # noqa: BLE001 - long-running daemon loop
            pass
        await asyncio.sleep(interval)
