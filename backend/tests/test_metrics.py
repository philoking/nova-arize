"""Unit tests for the HUD metrics aggregator.

The network clients (Beszel/Frigate) are stubbed, so these cover the pure
aggregation logic: the load-average series joined onto each host by Beszel record
id, the online-first ordering, and the shared TTL cache. Stdlib only.

Run from ``backend/``:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import metrics  # noqa: E402


class Aggregate(unittest.TestCase):
    def setUp(self):
        self._orig = (metrics.beszel.systems, metrics.beszel.load_history, metrics._frigate)
        metrics._cache = None

    def tearDown(self):
        metrics.beszel.systems, metrics.beszel.load_history, metrics._frigate = self._orig
        metrics._cache = None

    def _run(self, systems, frig=None, load=None):
        async def _s(): return systems
        async def _l(ids): return load or {}
        async def _f(): return frig
        metrics.beszel.systems = _s
        metrics.beszel.load_history = _l
        metrics._frigate = _f
        return asyncio.run(metrics.get_metrics(force=True))

    def test_every_beszel_host_is_reported(self):
        """No host list is curated server-side - whatever Beszel monitors shows up."""
        d = self._run(
            [{"id": "a", "name": "Nova Server", "status": "up", "cpu": 1, "mem": 2, "disk": 3},
             {"id": "b", "name": "Kali", "status": "up", "cpu": 0, "mem": 1, "disk": 9},
             {"id": "c", "name": "Shuri", "status": "up", "cpu": 4, "mem": 5, "disk": 6}])
        self.assertEqual({h["name"] for h in d["hosts"]}, {"Nova Server", "Kali", "Shuri"})
        self.assertNotIn("id", d["hosts"][0])   # internal Beszel id stays off the payload

    def test_load_series_joined_by_id(self):
        d = self._run([{"id": "a", "name": "Nova", "status": "up"},
                       {"id": "b", "name": "Kali", "status": "up"}],
                      load={"a": [0.5, 0.7]})
        by = {h["name"]: h for h in d["hosts"]}
        self.assertEqual(by["Nova"]["load"], [0.5, 0.7])
        self.assertNotIn("load", by["Kali"])     # no history → no sparkline

    def test_online_first_then_by_name(self):
        d = self._run(
            [{"name": "Zeta", "status": "up", "cpu": 0, "mem": 0, "disk": 0},
             {"name": "Alpha", "status": "down", "cpu": 0, "mem": 0, "disk": 0},
             {"name": "Beta", "status": "up", "cpu": 0, "mem": 0, "disk": 0}])
        self.assertEqual([h["name"] for h in d["hosts"]], ["Beta", "Zeta", "Alpha"])

    def test_frigate_passthrough(self):
        d = self._run([], frig={"cameras": 29})
        self.assertEqual(d["frigate"], {"cameras": 29})

    def test_cache_shared_until_forced(self):
        self._run([{"name": "X", "status": "up", "cpu": 0, "mem": 0, "disk": 0}])
        async def empty(): return []
        metrics.beszel.systems = empty                 # source changed…
        d = asyncio.run(metrics.get_metrics(force=False))  # …but cache still valid
        self.assertEqual([h["name"] for h in d["hosts"]], ["X"])


if __name__ == "__main__":
    unittest.main()
