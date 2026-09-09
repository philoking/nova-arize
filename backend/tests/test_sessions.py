"""Unit tests for server-owned sessions (#44).

The server is the source of truth for conversation context: a client sends only
the new user message + a ``conversation_id``; the backend rehydrates the stored
thread (``history.thread_for``) and appends the new turn. The key property is
*cross-surface continuity* - a thread started on the satellite continues with its
full prior context from the web, and vice-versa.

Stdlib ``unittest``. Run from ``backend/``:  python -m unittest discover -s tests
Uses a throwaway DB (set before importing the app).
"""

import os
import sys
import tempfile
import unittest

os.environ["NOVA_VOICE_HISTORY_DB"] = os.path.join(tempfile.mkdtemp(), "test_sessions.db")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import history  # noqa: E402


def assemble(conversation_id, new_user):
    """Mirror of the /api/chat context assembly (#44), for asserting behavior."""
    return history.thread_for(conversation_id) + [{"role": "user", "content": new_user}]


class ThreadFor(unittest.TestCase):
    def setUp(self):
        history.init()
        db = history._db()
        db.execute("DELETE FROM messages")
        db.execute("DELETE FROM conversations")
        db.commit()

    def test_new_or_unknown_conversation_is_empty(self):
        self.assertEqual(history.thread_for(None), [])
        self.assertEqual(history.thread_for("does-not-exist"), [])

    def test_returns_model_ready_role_content_dicts_in_order(self):
        cid = history.record_turn(None, "web", "hello", "hi there")
        history.record_turn(cid, "web", "what's 2+2", "four")
        self.assertEqual(
            history.thread_for(cid),
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "what's 2+2"},
                {"role": "assistant", "content": "four"},
            ],
        )

    def test_assembly_appends_only_the_new_user_turn(self):
        cid = history.record_turn(None, "web", "remember 7", "noted")
        msgs = assemble(cid, "what did I say")
        self.assertEqual(msgs[-1], {"role": "user", "content": "what did I say"})
        # Prior turns are replayed exactly once - no duplication of the new turn.
        self.assertEqual(sum(m["content"] == "what did I say" for m in msgs), 1)
        self.assertEqual(len(msgs), 3)  # user + assistant + new user

    def test_cross_surface_continuity(self):
        # Started on the satellite...
        cid = history.record_turn(None, "satellite", "my name is Jason", "got it")
        # ...continued from the web: the assembled context carries the satellite
        # turn even though the web client sent only the new message.
        msgs = assemble(cid, "what's my name")
        contents = [m["content"] for m in msgs]
        self.assertIn("my name is Jason", contents)
        self.assertIn("got it", contents)
        self.assertEqual(contents[-1], "what's my name")


if __name__ == "__main__":
    unittest.main()
