"""Unit tests for the auto-escalation struggle detector.

The detector decides when a local (qwen3) reply "struggled" and should be retried
on Claude. It must fire on empty answers and clear punts/deferrals (the observed
qwen3 failure modes) but NOT on genuine, substantive answers - a false positive
spends Claude tokens on a turn that was fine. Stdlib only.

Run from ``backend/``:  python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest

os.environ["NOVA_VOICE_HISTORY_DB"] = os.path.join(tempfile.mkdtemp(), "test_esc.db")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import escalation  # noqa: E402


class StruggleDetector(unittest.TestCase):
    def test_empty_reply_escalates(self):
        self.assertTrue(escalation.is_struggle(""))
        self.assertTrue(escalation.is_struggle("   \n "))

    def test_punts_escalate(self):
        for reply in [
            "I don't have access to a manual for the Rikon 10-3061 band saw.",
            "Would you like me to search the web for this information?",
            "I recommend checking the manufacturer's website for the exact blade.",
            "For precise specifications, consult the manufacturer directly.",
            "You'll need to check the manual for your specific model.",
            "I'm unable to find the blade size for that model.",
            "You can search online or contact Rikon customer support.",
        ]:
            self.assertTrue(escalation.is_struggle(reply), reply)

    def test_substantive_answers_do_not_escalate(self):
        for reply in [
            "Your Grizzly G0623X Sliding Table Saw uses a 10\" blade with a 5/8\" arbor.",
            "The Rikon 10-3061 uses a 70-1/2\" blade, 1/4\" wide, 6 TPI.",
            "I set a timer for 10 minutes.",
            "The capital of Chile is Santiago.",
            "Here are three ideas for dinner: pasta, tacos, or a stir-fry. Let me know if you'd like recipes.",
        ]:
            self.assertFalse(escalation.is_struggle(reply), reply)

    def test_benign_closer_is_not_a_struggle(self):
        # "let me know if you'd like more" is a friendly closer, not a deferral.
        self.assertFalse(escalation.is_struggle(
            "The blade is 10 inches. Let me know if you'd like torque specs too."))


class UpfrontIntent(unittest.TestCase):
    def test_research_and_writing_escalate(self):
        for msg in [
            "research the best cordless drills under $200 and summarize",
            "write me an email declining the meeting",
            "draft a project plan for the kitchen remodel",
            "explain quantum entanglement in detail",
            "give me a step-by-step guide to sharpening chisels",
            "compare the DeWalt and Milwaukee impact drivers",
            "what are the pros and cons of a sliding table saw?",
            "do a thorough analysis of my options",
            "brainstorm names for my woodworking channel",
        ]:
            self.assertTrue(escalation.should_escalate_upfront(msg), msg)

    def test_explicit_ask_claude_escalates(self):
        self.assertTrue(escalation.should_escalate_upfront("ask Claude what the tariff situation is"))
        self.assertTrue(escalation.should_escalate_upfront("use Claude for this one"))

    def test_simple_turns_stay_local(self):
        for msg in [
            "what's the capital of Chile?",
            "set a timer for 10 minutes",
            "turn off the office lights",
            "what blade does my grizzly table saw use?",
            "write a timer for 5 minutes",     # 'write' but no writing artifact
            "add milk to my shopping list",
            "remind me to call mom at 5pm",
        ]:
            self.assertFalse(escalation.should_escalate_upfront(msg), msg)


if __name__ == "__main__":
    unittest.main()
