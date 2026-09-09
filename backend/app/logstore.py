"""Verbose activity log - a small, capped SQLite ring buffer for troubleshooting.

There's no way to watch what the assistant is doing without SSHing to Nova and
tailing the container, so this captures the app's own log records into the shared
DB and exposes them at ``/api/log`` for a viewer in the Settings screen (#21).

It's a ``logging.Handler`` (``SqliteLogHandler``) attached to the root logger *and*
to uvicorn's own loggers, so the capture is deliberately verbose: our ``nova``
events, every HTTP request (uvicorn.access), server lifecycle (uvicorn), and the
outbound upstream calls (httpx's "HTTP Request:" lines) all land here. Only the
byte-level ``httpcore`` transport chatter is muted (and it's DEBUG anyway, so it
never reaches us at the INFO default).

Kept deliberately tiny and self-bounding: a row cap plus an age cutoff
(``log_retention_hours``), pruned at startup and every so often on insert, so the
buffer is a rolling window rather than an unbounded table. Same shared connection
/ ``/data`` volume as the history and timer stores.
"""

from __future__ import annotations

import contextvars
import logging
import time

from . import db
from .config import settings

# Set True while a health check is running so the upstream probes it fans out (the
# httpx "HTTP Request:" lines) are tagged as noise and can be filtered out of the
# viewer. See main.py's /api/health handler.
health_check_var: contextvars.ContextVar[bool] = contextvars.ContextVar("nova_health_check", default=False)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    level   TEXT NOT NULL,
    logger  TEXT NOT NULL,
    message TEXT NOT NULL,
    noise   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_logs_id ON logs(id DESC);
"""

# Only the byte-level transport chatter is dropped - everything else (requests,
# tool calls, upstream calls, errors) is captured so the log is fully verbose.
_MUTED = ("httpcore",)

# Routine background chatter the viewer can hide: health probes, presence/heartbeat,
# timer and alarm polling from the panel and satellites, config/voice fetches, and
# the upstream voice-list call behind them. Tagged noise=1 at write time so paging
# can filter server-side and still return meaningful rows.
_NOISE_PATHS = ("/api/health", "/api/config", "/api/timers", "/api/alarms",
                "/api/satellites", "/api/log")
# httpx "HTTP Request:" chatter that isn't tied to a health probe - the periodic
# voice-list fetch behind /api/config and the Settings screen.
_NOISE_UPSTREAM = ("/v1/audio/voices",)


def _is_polling_noise(logger: str, message: str) -> bool:
    if logger == "uvicorn.access":
        return any(p in message for p in _NOISE_PATHS)
    if logger == "httpx":
        return any(p in message for p in _NOISE_UPSTREAM)
    return False

_lock = db.lock
_ready = False
_installed = False
# Prune every N inserts so the table stays bounded without a background task.
_since_prune = 0
_PRUNE_EVERY = 200

# The app's own logger. Use this from route handlers to record meaningful events.
log = logging.getLogger("nova")


def _now() -> float:
    return time.time()


def init() -> None:
    """Create the schema on the shared connection. Idempotent; call at startup."""
    global _ready
    with _lock:
        c = db.conn()
        c.executescript(_SCHEMA)
        # Migrate an existing table that predates the `noise` column.
        cols = {r[1] for r in c.execute("PRAGMA table_info(logs)").fetchall()}
        if "noise" not in cols:
            c.execute("ALTER TABLE logs ADD COLUMN noise INTEGER NOT NULL DEFAULT 0")
        c.commit()
        _ready = True


def _db():
    if not _ready:
        init()
    return db.conn()


def record(ts: float, level: str, logger: str, message: str, noise: bool = False) -> None:
    """Append one log line, pruning periodically to keep the buffer bounded."""
    global _since_prune
    with _lock:
        _db().execute(
            "INSERT INTO logs (ts, level, logger, message, noise) VALUES (?, ?, ?, ?, ?)",
            (ts, level, logger, message, 1 if noise else 0),
        )
        _db().commit()
        _since_prune += 1
        if _since_prune >= _PRUNE_EVERY:
            _since_prune = 0
            _prune_locked()


def recent(limit: int = 500, since: int = 0, before: int = 0,
           level: str | None = None, hide_noise: bool = False) -> list[dict]:
    """Newest-first log lines from the rolling 48h store.

    ``since`` returns only rows newer than that id (incremental polling for new
    activity); ``before`` returns only rows older than that id (paging back
    through history); ``level`` filters to that level and above; ``hide_noise``
    drops the routine health/polling chatter.
    """
    limit = max(1, min(limit, 2000))
    clauses: list[str] = []
    params: list = []
    if hide_noise:
        clauses.append("noise = 0")
    if since:
        clauses.append("id > ?")
        params.append(since)
    if before:
        clauses.append("id < ?")
        params.append(before)
    if level:
        wanted = _at_or_above(level)
        if wanted:
            marks = ",".join("?" * len(wanted))
            clauses.append(f"level IN ({marks})")
            params.extend(wanted)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _lock:
        rows = _db().execute(
            f"SELECT id, ts, level, logger, message FROM logs {where} "
            f"ORDER BY id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [dict(r) for r in rows]


_ORDER = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _at_or_above(level: str) -> list[str]:
    try:
        i = _ORDER.index(level.upper())
    except ValueError:
        return []
    return _ORDER[i:]


def _prune_locked() -> int:
    """Drop rows past the age cutoff or beyond the row cap. Caller holds the lock."""
    d = _db()
    cutoff = _now() - settings.log_retention_hours * 3600
    d.execute("DELETE FROM logs WHERE ts < ?", (cutoff,))
    # Cap the row count: delete everything older than the newest `log_max_rows`.
    d.execute(
        "DELETE FROM logs WHERE id NOT IN "
        "(SELECT id FROM logs ORDER BY id DESC LIMIT ?)",
        (settings.log_max_rows,),
    )
    d.commit()
    return d.total_changes


def prune() -> int:
    with _lock:
        return _prune_locked()


def reclassify_noise() -> int:
    """Backfill the ``noise`` flag on existing rows against the current rules.

    Run once at startup so a rule change (e.g. adding ``/api/alarms``) applies to
    already-stored rows too, not just new ones - otherwise the last 48h of polling
    would keep showing until the buffer rolled. Cheap; the paths carry no LIKE
    wildcards so the ``%…%`` match is safe."""
    with _lock:
        d = _db()
        for p in _NOISE_PATHS:
            d.execute("UPDATE logs SET noise=1 WHERE noise=0 AND logger='uvicorn.access' "
                      "AND message LIKE ?", (f"%{p}%",))
        for p in _NOISE_UPSTREAM:
            d.execute("UPDATE logs SET noise=1 WHERE noise=0 AND logger='httpx' "
                      "AND message LIKE ?", (f"%{p}%",))
        d.commit()
        return d.total_changes


def clear() -> None:
    """Empty the log buffer (the viewer's Clear button)."""
    with _lock:
        _db().execute("DELETE FROM logs")
        _db().commit()


class SqliteLogHandler(logging.Handler):
    """Routes log records into the SQLite ring buffer. Never raises into logging."""

    def emit(self, rec: logging.LogRecord) -> None:  # noqa: D401
        try:
            if rec.name in _MUTED or rec.name.startswith("httpcore"):
                return
            msg = rec.getMessage()
            if rec.exc_info:
                # Append the traceback so errors are actionable in the viewer.
                msg = msg + "\n" + logging.Formatter().formatException(rec.exc_info)
            # Noise = an upstream call made during a health check, or an access
            # line for one of the routine polling/health endpoints.
            noise = health_check_var.get() or _is_polling_noise(rec.name, msg)
            record(getattr(rec, "created", None) or _now(), rec.levelname, rec.name, msg, noise=noise)
        except Exception:  # noqa: BLE001 - logging must never break the app
            self.handleError(rec)


def install() -> None:
    """Attach the handler so the app's activity is captured. Idempotent.

    Goes on the root logger (our ``nova`` events, httpx, and anything that
    propagates) plus uvicorn's ``uvicorn`` and ``uvicorn.access`` loggers, which
    uvicorn configures NOT to propagate - so every HTTP request is logged too.
    ``uvicorn.error`` is skipped here because it already propagates to root
    (attaching would double-log it).
    """
    global _installed
    if _installed:
        return
    init()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = SqliteLogHandler(level=level)
    for name in ("", "uvicorn", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        # Ensure the logger's own threshold lets our level through.
        if lg.level == logging.NOTSET or lg.level > level:
            lg.setLevel(level)
    _installed = True
