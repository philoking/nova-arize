"""An empty web search must distinguish "nothing matched" from "search is down".

SearXNG returns HTTP 200 with `results: []` in both cases; only
`unresponsive_engines` separates them. Reporting both as "no results found" told
the model the web held nothing on the subject, so it answered from its own
knowledge and presented that as a search-informed reply - an ungrounded answer
indistinguishable from a grounded one, in the reply *and* in the trace.

Found while investigating why a traced turn said "web search is coming up empty".
Every engine was CAPTCHA'd or rate-limited. Stdlib only.

Run from ``backend/``:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import searxng  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, *a, **kw):
        return _FakeResponse(self._payload)


class EmptyResults(unittest.TestCase):
    def _search(self, payload):
        orig = searxng._client
        searxng._client = lambda: _FakeClient(payload)
        try:
            return asyncio.run(searxng.search("dust collection cfm"))
        finally:
            searxng._client = orig

    def test_genuine_no_results_is_a_note(self):
        out = self._search({"results": [], "unresponsive_engines": []})
        self.assertEqual(out.get("note"), "no results found")
        self.assertNotIn("error", out)

    def test_all_engines_down_is_an_error(self):
        """The regression: this used to be reported as 'no results found'."""
        out = self._search({"results": [], "unresponsive_engines": [
            ["brave", "Suspended: too many requests"],
            ["duckduckgo", "CAPTCHA"],
            ["google", "Suspended: CAPTCHA"],
        ]})
        self.assertIn("error", out)
        self.assertNotIn("note", out)
        self.assertIn("brave", out["error"])
        self.assertEqual(out["unresponsive_engines"], ["brave", "duckduckgo", "google"])

    def test_error_tells_the_model_not_to_answer_anyway(self):
        out = self._search({"results": [], "unresponsive_engines": [["google", "CAPTCHA"]]})
        self.assertIn("NOT an absence of results", out["error"])

    def test_results_present_ignores_partial_engine_failures(self):
        """Some engines failing while others returned hits is normal, not an error."""
        out = self._search({
            "results": [{"title": "T", "url": "u", "content": "c"}],
            "unresponsive_engines": [["google", "CAPTCHA"]],
        })
        self.assertNotIn("error", out)
        self.assertEqual(len(out["results"]), 1)

    def test_instant_answer_is_not_an_error(self):
        out = self._search({
            "results": [], "answers": ["42"],
            "unresponsive_engines": [["google", "CAPTCHA"]],
        })
        self.assertNotIn("error", out)
        self.assertEqual(out["answers"], ["42"])
