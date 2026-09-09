"""Unit tests for capability routing (#35/#39).

Covers the pure routing helpers: deterministic keyword matching (including the
query that regressed #26), router-reply parsing, and skill availability. Stdlib
``unittest`` - no DB or network.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import skills  # noqa: E402

ALL = list(skills.SKILLS)


class Deterministic(unittest.TestCase):
    def m(self, msg, extra=None):
        return skills.deterministic_match(msg, ALL, extra)

    def test_tasks_query_that_regressed_26(self):
        self.assertEqual(self.m("read me today's tasks"), ["tasks"])
        self.assertEqual(self.m("do I have any open tasks from yesterday"), ["tasks"])

    def test_lists(self):
        self.assertIn("lists", self.m("add milk to my shopping list"))
        self.assertIn("lists", self.m("what's on my grocery list"))

    def test_timers(self):
        self.assertEqual(self.m("set an alarm for 7am"), ["timers"])
        self.assertEqual(self.m("remind me in ten minutes"), ["timers"])

    def test_home_matches_via_dynamic_device_name(self):
        self.assertEqual(self.m("turn on the office lamp", {"home": ["office lamp"]}), ["home"])

    def test_lock_status_and_control_route_home(self):
        # Inflected lock forms must route to 'home' - whole-word matching gives no
        # stemming, so "which doors are locked?" missed the bare "lock" keyword and
        # ran tool-free, punting instead of checking the locks.
        self.assertEqual(self.m("which doors are locked?"), ["home"])
        self.assertEqual(self.m("are any doors still unlocked"), ["home"])
        self.assertEqual(self.m("lock the shop door"), ["home"])

    def test_memory(self):
        self.assertEqual(self.m("remember I am vegetarian"), ["memory"])

    def test_web(self):
        self.assertIn("web", self.m("look up the weather in Boston"))

    def test_general_knowledge_matches_nothing(self):
        self.assertEqual(self.m("what is the capital of France"), [])
        self.assertEqual(self.m("tell me a joke"), [])

    def test_whole_word_matching_avoids_false_positive(self):
        # 'list' must not match inside 'listen'; nothing here is a skill trigger.
        self.assertEqual(self.m("listen to this idea"), [])


class Availability(unittest.TestCase):
    def test_only_skills_with_enabled_tools(self):
        names = {"notes__list_tasks", "notes__add_to_list", "timers__start_timer"}
        got = {s.name for s in skills.available(names)}
        self.assertIn("tasks", got)
        self.assertIn("lists", got)
        self.assertIn("timers", got)
        self.assertNotIn("home", got)   # no HA tools present
        self.assertNotIn("web", got)    # no web tools present
        self.assertNotIn("memory", got)

    def test_empty_when_nothing_enabled(self):
        self.assertEqual(skills.available(set()), [])


class IsFollowup(unittest.TestCase):
    def test_pronoun_and_continuation_cues(self):
        self.assertTrue(skills.is_followup("please turn them back off"))
        self.assertTrue(skills.is_followup("they are still on"))
        self.assertTrue(skills.is_followup("do it again"))

    def test_a_fresh_request_is_not_a_followup(self):
        self.assertFalse(skills.is_followup("what is the capital of France"))
        self.assertFalse(skills.is_followup("add milk to my shopping list"))


class _FakeHome:
    """Minimal HA-like provider so Registry.route has the 'home' skill available."""
    name = "home_assistant"
    enabled = True

    def tool_specs(self):
        return [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {}}}}
                for n in ("find_entities", "get_state", "call_service")]

    async def execute(self, tool, args):
        return {}

    def system_note(self):
        return " ha"

    async def health(self):
        return True


class FollowupRouting(unittest.TestCase):
    def setUp(self):
        from app.plugins.registry import Registry
        self.r = Registry([_FakeHome()])

    def test_direct_command_routes_home(self):
        self.assertEqual(self.r.route(["can you turn on the living room lights"]), ["home"])

    def test_pronoun_followup_inherits_home(self):
        # The exact sequence that regressed: 'turn them back off' must reach call_service.
        seq = ["can you turn on the living room lights", "please turn them back off"]
        self.assertEqual(self.r.route(seq), ["home"])

    def test_followup_walks_back_past_an_unmatched_turn(self):
        seq = ["turn on the living room lights", "please turn them back off", "they are still on"]
        self.assertEqual(self.r.route(seq), ["home"])

    def test_fresh_topic_does_not_inherit(self):
        seq = ["turn on the living room lights", "what is the capital of France"]
        self.assertEqual(self.r.route(seq), [])


if __name__ == "__main__":
    unittest.main()
