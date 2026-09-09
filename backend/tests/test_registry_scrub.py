"""Unit tests for the tool-argument scrubber (#19).

Stdlib ``unittest`` - no test dependency to install. Run from ``backend/``:

    python -m unittest discover -s tests
"""

import os
import sys
import unittest

# Make the app package importable when run from backend/ or the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.plugins.registry import _scrub  # noqa: E402


class ScrubToolArgs(unittest.TestCase):
    def test_strips_leading_no_think(self):
        self.assertEqual(_scrub({"content": "/no_think buy milk"}), {"content": "buy milk"})

    def test_strips_standalone_token(self):
        self.assertEqual(_scrub("/no_think"), "")
        self.assertEqual(_scrub("/think"), "")

    def test_strips_think_block(self):
        self.assertEqual(_scrub("<think>reasoning</think>Take vitamins"), "Take vitamins")

    def test_strips_token_in_the_middle(self):
        self.assertEqual(_scrub("call mom /no_think tonight"), "call mom tonight")

    def test_leaves_clean_values_untouched(self):
        # No control tokens → returned byte-for-byte, including intentional spacing.
        for v in ("  spaced  ", "rethink the plan", "author: nothink", "line1\nline2"):
            self.assertEqual(_scrub(v), v)

    def test_word_boundary_avoids_false_positives(self):
        # "/no_thinking" isn't the control token - must not be stripped.
        self.assertEqual(_scrub("/no_thinking about it"), "/no_thinking about it")

    def test_recurses_into_dicts_and_lists(self):
        arg = {"items": ["/no_think eggs", "bread"], "note": {"body": "/think ham"}}
        self.assertEqual(_scrub(arg), {"items": ["eggs", "bread"], "note": {"body": "ham"}})

    def test_non_strings_pass_through(self):
        self.assertEqual(_scrub({"count": 3, "done": True, "ratio": 1.5}),
                         {"count": 3, "done": True, "ratio": 1.5})


if __name__ == "__main__":
    unittest.main()
