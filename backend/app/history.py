"""Conversation history - a small SQLite store shared by every surface.

Every turn from the browser *and* the satellite is persisted here under a
``conversation_id``, so past chats are listable and a conversation started on one
surface (e.g. the Pi satellite) can be revisited or continued on another (the web
client). Kept deliberately tiny: stdlib ``sqlite3`` (no new dependency), one
connection guarded by a lock, synchronous calls (each is sub-millisecond at this
single-user scale). History older than ``history_days`` is pruned so the log
stays a rolling window rather than growing forever.

The DB file lives on the same ``data/`` dir / mounted volume as the settings
file - see ``config.py``.
"""

from __future__ import annotations

import json
import time
import uuid

from . import db
from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    tool_calls      TEXT,
    escalated       INTEGER NOT NULL DEFAULT 0,   -- 1 if this reply came from Claude
    model           TEXT                          -- model that generated the reply
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
"""

_TITLE_MAX = 60

# Alias the shared lock so the existing `with _lock:` transactions keep working.
_lock = db.lock
_ready = False


def _now() -> int:
    return int(time.time())


def _title_from(text: str) -> str:
    """A short, human label for a conversation from its first user message."""
    line = " ".join((text or "").split())
    if not line:
        return "New conversation"
    return line[:_TITLE_MAX] + ("…" if len(line) > _TITLE_MAX else "")


def init() -> None:
    """Create the schema on the shared connection. Idempotent; call at startup."""
    global _ready
    with db.lock:
        conn = db.conn()
        conn.executescript(_SCHEMA)
        # Migration: `tool_calls` was added after the table shipped, so an existing
        # DB won't have it (CREATE TABLE IF NOT EXISTS is a no-op there). Add it.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "tool_calls" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN tool_calls TEXT")
        if "escalated" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN escalated INTEGER NOT NULL DEFAULT 0")
        if "model" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN model TEXT")
        conn.commit()
        _ready = True


def _db():
    if not _ready:
        init()
    return db.conn()


def record_turn(conversation_id: str | None, source: str, user_text: str, assistant_text: str,
                tool_calls: list[dict] | None = None, escalated: bool = False,
                model: str | None = None) -> str:
    """Append one user+assistant exchange, creating the conversation if needed.

    Returns the conversation id (freshly minted when ``conversation_id`` is None
    or unknown). Only the new turn is stored - callers pass the full history to
    the model but persist just what's new here, so nothing is duplicated.
    ``tool_calls`` (``[{"tool", "args"}, …]``) are the tools the agent ran this
    turn, stored on the assistant row so the UI can re-render its action chips
    when the conversation is reopened.
    """
    cid = conversation_id or uuid.uuid4().hex
    now = _now()
    with _lock:
        db = _db()
        # Create on first sighting; the title is set once, from the opening line.
        db.execute(
            "INSERT OR IGNORE INTO conversations (id, title, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cid, _title_from(user_text), source, now, now),
        )
        if user_text:
            db.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                (cid, user_text, now),
            )
        if assistant_text:
            db.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at, tool_calls, escalated, model) "
                "VALUES (?, 'assistant', ?, ?, ?, ?, ?)",
                (cid, assistant_text, now, json.dumps(tool_calls) if tool_calls else None,
                 1 if escalated else 0, model),
            )
        db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))
        db.commit()
    return cid


def list_conversations(days: int | None = None) -> list[dict]:
    """Conversations active within the window, most recently used first."""
    window = settings.history_days if days is None else days
    cutoff = _now() - window * 86400
    with _lock:
        rows = _db().execute(
            "SELECT c.id, c.title, c.source, c.created_at, c.updated_at, "
            "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
            "FROM conversations c WHERE c.updated_at >= ? ORDER BY c.updated_at DESC",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: str) -> dict | None:
    """A conversation with its full ordered message list, or None if unknown."""
    with _lock:
        db = _db()
        conv = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if conv is None:
            return None
        msgs = db.execute(
            "SELECT role, content, created_at, tool_calls, escalated, model FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    result = dict(conv)
    result["messages"] = [_shape_message(m) for m in msgs]
    return result


def _shape_message(row) -> dict:
    """Row → API dict, parsing the stored ``tool_calls`` JSON back into a list
    (``[]`` when the turn ran no tools or predates the column)."""
    m = dict(row)
    raw = m.pop("tool_calls", None)
    try:
        m["tool_calls"] = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        m["tool_calls"] = []
    m["escalated"] = bool(m.get("escalated"))
    return m


def thread_for(conversation_id: str | None) -> list[dict]:
    """The stored turns of a conversation as model-ready ``{role, content}`` dicts.

    This is what the server replays as prior context for a new turn (#44): the
    shared history store is the source of truth, so a thread started on one
    surface (e.g. the satellite) continues with its full context on another (the
    web client). Empty for a new or unknown conversation.
    """
    conv = get_conversation(conversation_id) if conversation_id else None
    if not conv:
        return []
    return [{"role": m["role"], "content": m["content"]} for m in conv["messages"]]


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and its messages. True if it existed."""
    with _lock:
        db = _db()
        cur = db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        db.commit()
        return cur.rowcount > 0


def prune(days: int | None = None) -> int:
    """Drop conversations (and their messages) older than the window. Returns the
    number removed. Cheap enough to run at startup and after each new turn."""
    window = settings.history_days if days is None else days
    cutoff = _now() - window * 86400
    with _lock:
        db = _db()
        stale = [r["id"] for r in db.execute(
            "SELECT id FROM conversations WHERE updated_at < ?", (cutoff,)).fetchall()]
        if stale:
            marks = ",".join("?" * len(stale))
            db.execute(f"DELETE FROM messages WHERE conversation_id IN ({marks})", stale)
            db.execute(f"DELETE FROM conversations WHERE id IN ({marks})", stale)
            db.commit()
        return len(stale)
