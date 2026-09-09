"""Unit tests for list remove/clear (#14).

Exercises ``remove_item`` / ``clear_list`` against a fake Obsidian REST client, so
no live vault is needed. Stdlib ``unittest`` + ``asyncio``.
Run from ``backend/``:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest

os.environ.setdefault("NOVA_OBSIDIAN_URL", "http://fake.local")
os.environ.setdefault("NOVA_OBSIDIAN_API_KEY", "x")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import obsidian  # noqa: E402


class _Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text


class _FakeClient:
    """Serves one note at ``path`` and records the last PUT body."""
    def __init__(self, path, text):
        self.url = obsidian._vault(path)
        self.text = text
        self.put_body = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, params=None):
        return _Resp(200, self.text) if url == self.url else _Resp(404)

    async def put(self, url, content=None, headers=None):
        self.put_body = content
        return _Resp(200)


def run(coro):
    return asyncio.run(coro)


def with_note(text, path="Groceries.md"):
    """Patch obsidian._client to serve one note; return the fake for assertions."""
    fake = _FakeClient(path, text)
    obsidian._client = lambda: fake
    return fake


SHOPPING = "# Groceries\n\n- [ ] Milk\n- [x] Eggs\n- [ ] Red apples\n- [ ] Green apples\n"


class RemoveItem(unittest.TestCase):
    def test_removes_the_matching_line(self):
        fake = with_note(SHOPPING)
        res = run(obsidian.remove_item("Groceries", "milk"))
        self.assertTrue(res.get("ok"))
        self.assertEqual(res["removed"], "Milk")
        self.assertNotIn("Milk", fake.put_body)
        self.assertIn("Eggs", fake.put_body)  # others untouched

    def test_removes_a_checked_item_too(self):
        fake = with_note(SHOPPING)
        res = run(obsidian.remove_item("Groceries", "eggs"))
        self.assertEqual(res["removed"], "Eggs")
        self.assertNotIn("- [x] Eggs", fake.put_body)

    def test_ambiguous_returns_candidates_without_writing(self):
        fake = with_note(SHOPPING)
        res = run(obsidian.remove_item("Groceries", "apples"))
        self.assertEqual(res.get("error"), "ambiguous")
        self.assertCountEqual(res["candidates"], ["Red apples", "Green apples"])
        self.assertIsNone(fake.put_body)  # nothing removed on ambiguity

    def test_no_match_errors_without_writing(self):
        fake = with_note(SHOPPING)
        res = run(obsidian.remove_item("Groceries", "bananas"))
        self.assertIn("no item matching", res.get("error", ""))
        self.assertIsNone(fake.put_body)


class ClearList(unittest.TestCase):
    def test_clear_all_keeps_heading(self):
        fake = with_note(SHOPPING)
        res = run(obsidian.clear_list("Groceries"))
        self.assertEqual(res["removed"], 4)
        self.assertIn("# Groceries", fake.put_body)
        self.assertNotIn("- [", fake.put_body)

    def test_clear_checked_only(self):
        fake = with_note(SHOPPING)
        res = run(obsidian.clear_list("Groceries", checked_only=True))
        self.assertEqual(res["removed"], 1)
        self.assertNotIn("Eggs", fake.put_body)     # the checked one is gone
        self.assertIn("- [ ] Milk", fake.put_body)  # unchecked ones stay

    def test_clear_empty_list_errors(self):
        fake = with_note("# Groceries\n\n")
        res = run(obsidian.clear_list("Groceries"))
        self.assertIn("already empty", res.get("error", ""))
        self.assertIsNone(fake.put_body)


if __name__ == "__main__":
    unittest.main()
