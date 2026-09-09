"""Unit tests for the persistent-memory store (#29).

Stdlib ``unittest``. Run from ``backend/``:  python -m unittest discover -s tests
Uses a throwaway DB (set before importing the app).
"""

import os
import sys
import tempfile
import unittest

os.environ["NOVA_VOICE_HISTORY_DB"] = os.path.join(tempfile.mkdtemp(), "test_memory.db")
os.environ.setdefault("NOVA_VOICE_MEMORY_MID_DAYS", "30")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import memory  # noqa: E402


class MemoryStore(unittest.TestCase):
    def setUp(self):
        memory.init()
        memory._db().execute("DELETE FROM memories")
        memory._db().commit()

    def test_long_never_expires(self):
        m = memory.add("allergic to peanuts", tier="long")
        self.assertIsNone(m["expires_at"])

    def test_mid_expires_in_about_30_days(self):
        m = memory.add("on a diet", tier="mid")
        days = (m["expires_at"] - memory._now()) / 86400
        self.assertTrue(29 < days <= 30, days)

    def test_short_expires_end_of_today(self):
        m = memory.add("working from home", tier="short")
        # Expires later today (or this second, if run at 23:59:59) - always ≤ 24h.
        self.assertIsNotNone(m["expires_at"])
        self.assertLessEqual(m["expires_at"] - memory._now(), 86400)

    def test_unknown_tier_defaults_to_long(self):
        m = memory.add("prefers metric", tier="bogus")
        self.assertEqual(m["tier"], "long")

    def test_list_active_orders_long_then_mid_then_short(self):
        memory.add("today thing", tier="short")
        memory.add("month thing", tier="mid")
        memory.add("forever thing", tier="long")
        tiers = [m["tier"] for m in memory.list_active()]
        self.assertEqual(tiers, ["long", "mid", "short"])

    def test_prune_drops_expired(self):
        m = memory.add("stale", tier="mid")
        # Force it into the past.
        memory._db().execute("UPDATE memories SET expires_at=? WHERE id=?", (memory._now() - 1, m["id"]))
        memory._db().commit()
        memory.add("fresh", tier="long")
        removed = memory.prune()
        self.assertEqual(removed, 1)
        self.assertEqual([m["text"] for m in memory.list_active()], ["fresh"])

    def test_list_active_excludes_expired(self):
        m = memory.add("gone", tier="short")
        memory._db().execute("UPDATE memories SET expires_at=? WHERE id=?", (memory._now() - 1, m["id"]))
        memory._db().commit()
        self.assertEqual(memory.list_active(), [])

    def test_update_recomputes_expiry_from_new_tier(self):
        m = memory.add("temp", tier="short")
        self.assertIsNotNone(m["expires_at"])
        updated = memory.update(m["id"], tier="long")
        self.assertIsNone(updated["expires_at"])

    def test_forget_matching_is_case_insensitive_substring(self):
        memory.add("allergic to Peanuts", tier="long")
        memory.add("likes coffee", tier="long")
        removed = memory.forget_matching("peanut")
        self.assertEqual([m["text"] for m in removed], ["allergic to Peanuts"])
        self.assertEqual([m["text"] for m in memory.list_active()], ["likes coffee"])

    def test_delete_by_id(self):
        m = memory.add("x", tier="long")
        self.assertTrue(memory.delete(m["id"]))
        self.assertFalse(memory.delete(m["id"]))


if __name__ == "__main__":
    unittest.main()
