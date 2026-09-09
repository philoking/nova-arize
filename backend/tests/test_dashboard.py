"""Unit tests for the link dashboard store (Homepage replacement).

The store persists a user-edited groups/tiles document and validates every save so
a malformed client payload can't corrupt the file. These cover the sanitizer (the
security-relevant part) and the seed/persist/reset lifecycle. Stdlib only.

Run from ``backend/``:  python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import dashboard  # noqa: E402


class Sanitize(unittest.TestCase):
    def test_drops_empty_groups_and_tiles(self):
        out = dashboard._sanitize({"groups": [
            {"title": "", "tiles": [{"name": "x"}]},            # no title → dropped
            {"title": "Keep", "tiles": [
                {"name": "A", "href": "http://a"},
                {"junk": 1},                                    # no name/href → dropped
                {"href": "http://b"},                           # href-only kept
            ]},
        ]})
        self.assertEqual([g["title"] for g in out["groups"]], ["Keep"])
        self.assertEqual(len(out["groups"][0]["tiles"]), 2)

    def test_fills_and_dedups_ids(self):
        out = dashboard._sanitize({"groups": [
            {"title": "Dup", "tiles": [{"name": "Same"}, {"name": "Same"}]},
            {"title": "Dup", "tiles": []},
        ]})
        gids = [g["id"] for g in out["groups"]]
        self.assertEqual(len(set(gids)), 2)                     # group ids de-duplicated
        tids = [t["id"] for t in out["groups"][0]["tiles"]]
        self.assertEqual(len(set(tids)), 2)                     # tile ids de-duplicated
        self.assertTrue(all(t["id"] for t in out["groups"][0]["tiles"]))

    def test_metric_hint_kept_only_for_known_types(self):
        out = dashboard._sanitize({"groups": [{"title": "G", "tiles": [
            {"name": "Beszel", "href": "h", "metric": {"type": "beszel"}},
            {"name": "Evil", "href": "h", "metric": {"type": "rm -rf"}},   # unknown → stripped
        ]}]})
        tiles = out["groups"][0]["tiles"]
        self.assertEqual(tiles[0]["metric"], {"type": "beszel"})
        self.assertNotIn("metric", tiles[1])

    def test_never_raises_on_garbage(self):
        for junk in [None, {}, {"groups": "nope"}, {"groups": [None, 5, "x"]}, [1, 2, 3]]:
            self.assertEqual(dashboard._sanitize(junk), {"version": 1, "groups": []})


class StoreLifecycle(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "dash.json")
        self.store = dashboard.DashboardStore(self.path)

    def test_serves_seed_until_saved(self):
        seed = self.store.get()
        self.assertFalse(os.path.exists(self.path))            # nothing written yet
        self.assertGreater(len(seed["groups"]), 0)             # packaged seed is non-empty

    def test_save_persists_then_reset_reverts(self):
        self.store.save({"groups": [{"title": "Mine", "tiles": [{"name": "X", "href": "http://x"}]}]})
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual([g["title"] for g in self.store.get()["groups"]], ["Mine"])
        self.store.reset()
        self.assertFalse(os.path.exists(self.path))
        self.assertGreater(len(self.store.get()["groups"]), 1)  # back to the seed


if __name__ == "__main__":
    unittest.main()
