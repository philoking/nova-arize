"""Shared SQLite connection for the small on-disk stores (history + timers).

One connection guarded by one re-entrant lock, so the async app can use it safely
across threads. WAL mode plus a busy timeout keep the brief writes from the
request path and the timer scheduler from stepping on each other. The file is
``config.history_db`` - a single DB on the ``/data`` volume holds all local state.
"""

from __future__ import annotations

import os
import sqlite3
import threading

from .config import settings

# Re-entrant so a helper that already holds the lock can call another that takes
# it without deadlocking.
lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    """The shared connection, opened (and PRAGMA-tuned) on first use."""
    global _conn
    if _conn is None:
        with lock:
            if _conn is None:
                os.makedirs(os.path.dirname(settings.history_db) or ".", exist_ok=True)
                c = sqlite3.connect(settings.history_db, check_same_thread=False)
                c.row_factory = sqlite3.Row
                c.execute("PRAGMA foreign_keys = ON")
                c.execute("PRAGMA journal_mode = WAL")
                c.execute("PRAGMA busy_timeout = 5000")
                _conn = c
    return _conn
