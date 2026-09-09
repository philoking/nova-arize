"""The `thinking` parameter must match what each Claude model accepts.

Adaptive thinking is a 4.6-generation feature. Sending it to claude-haiku-4-5 -
this app's FAST escalation tier - returns a 400 and the turn fails outright.
That shipped unnoticed because only the *reactive* escalation path uses the fast
tier, so the failure never appeared on the upfront (Opus) path: qwen3 would
struggle, the retry would 400, and the user got an error instead of an answer.
Found by tracing the turn, not by reading the code. Stdlib only.

Run from ``backend/``:  python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.claude import _thinking_for  # noqa: E402


class ThinkingParam(unittest.TestCase):
    def test_deep_tier_gets_adaptive(self):
        for model in ("claude-opus-4-8", "claude-opus-5", "claude-opus-4-7", "claude-sonnet-5"):
            self.assertEqual(_thinking_for(model), {"type": "adaptive"}, model)

    def test_haiku_gets_none(self):
        """The regression: haiku 4.5 rejects adaptive thinking with a 400."""
        self.assertIsNone(_thinking_for("claude-haiku-4-5"))

    def test_older_models_get_none(self):
        self.assertIsNone(_thinking_for("claude-3-5-sonnet-20241022"))
        self.assertIsNone(_thinking_for("claude-sonnet-4-5"))

    def test_case_insensitive(self):
        self.assertIsNone(_thinking_for("Claude-Haiku-4-5"))

    def test_unknown_model_defaults_to_adaptive(self):
        """Unknown/future models get adaptive - the configured defaults are all
        modern. A future model that rejects it must be added to the list."""
        self.assertEqual(_thinking_for("claude-something-new"), {"type": "adaptive"})

    def test_missing_model(self):
        self.assertEqual(_thinking_for(""), {"type": "adaptive"})
        self.assertEqual(_thinking_for(None), {"type": "adaptive"})
