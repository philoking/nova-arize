"""Unit tests for MQTT notification gating (#16).

Covers the pure decision logic - global toggle, per-category allowlist, and the
timer→category mapping - without touching the network. ``aiomqtt`` is imported
lazily inside ``_publish_raw``, so these tests need no broker and no dependency.

Stdlib ``unittest``. Run from ``backend/``:  python -m unittest discover -s tests
Uses a throwaway settings file (set before importing the app).
"""

import os
import sys
import tempfile
import unittest

os.environ["NOVA_VOICE_SETTINGS_FILE"] = os.path.join(tempfile.mkdtemp(), "test_settings.json")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import notify  # noqa: E402
from app.settings_store import store  # noqa: E402


class Gating(unittest.TestCase):
    def setUp(self):
        # Start from a known state each test.
        for k in ("notifications_enabled", "notification_categories"):
            store.reset(k)

    def test_disabled_globally_blocks_all(self):
        store.set("notifications_enabled", "0")
        store.set("notification_categories", "reminders,timers,alarms")
        self.assertFalse(notify.is_enabled("reminders"))

    def test_enabled_only_for_selected_categories(self):
        store.set("notifications_enabled", "1")
        store.set("notification_categories", "reminders")
        self.assertTrue(notify.is_enabled("reminders"))
        self.assertFalse(notify.is_enabled("timers"))
        self.assertFalse(notify.is_enabled("alarms"))

    def test_categories_parse_ignores_whitespace_and_blanks(self):
        store.set("notifications_enabled", "1")
        store.set("notification_categories", " reminders , , timers ")
        self.assertEqual(notify.enabled_categories(), {"reminders", "timers"})

    def test_empty_categories_means_nothing_enabled(self):
        store.set("notifications_enabled", "1")
        store.set("notification_categories", "")
        self.assertFalse(notify.is_enabled("reminders"))


class CategoryMapping(unittest.TestCase):
    def test_alarm_wins_over_kind(self):
        self.assertEqual(notify.category_for({"is_alarm": 1, "kind": "reminder"}), "alarms")

    def test_timer_kind(self):
        self.assertEqual(notify.category_for({"is_alarm": 0, "kind": "timer"}), "timers")

    def test_reminder_default(self):
        self.assertEqual(notify.category_for({"is_alarm": 0, "kind": "reminder"}), "reminders")
        self.assertEqual(notify.category_for({}), "reminders")


class PublishRespectsGate(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        for k in ("notifications_enabled", "notification_categories", "mqtt_address", "mqtt_topic"):
            store.reset(k)

    async def test_publish_short_circuits_when_disabled(self):
        # Disabled → returns False without ever touching aiomqtt/network.
        store.set("notifications_enabled", "0")
        self.assertFalse(await notify.publish("reminders", "t", "m"))

    async def test_publish_enabled_but_no_broker_returns_false(self):
        # Enabled + category on, but no address/topic → warns, returns False (no raise).
        store.set("notifications_enabled", "1")
        store.set("notification_categories", "reminders")
        store.set("mqtt_address", "")
        store.set("mqtt_topic", "")
        self.assertFalse(await notify.publish("reminders", "t", "m"))


if __name__ == "__main__":
    unittest.main()
