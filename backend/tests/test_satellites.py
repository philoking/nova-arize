"""Unit tests for the satellite registry - in particular the sticky display name.

A satellite heartbeats every couple of seconds with its self-reported name, so a
name set from the web panel must NOT be clobbered by the next check-in. That's the
whole point of the separate `display_name` column; these tests lock it in. Stdlib
only; uses a throwaway DB like the other store tests.

Run from ``backend/``:  python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest

os.environ["NOVA_VOICE_HISTORY_DB"] = os.path.join(tempfile.mkdtemp(), "test_sat.db")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import satellites  # noqa: E402


class DisplayName(unittest.TestCase):
    def setUp(self):
        # Unique id per test + cleanup, so the shared connection (which a sibling
        # test module may have opened against a different temp DB) never collides.
        self.sid = f"sat-{self._testMethodName}"
        self.addCleanup(satellites.forget, self.sid)

    def test_defaults_to_device_name(self):
        satellites.heartbeat(self.sid, "raspberry-pi")
        s = satellites.get(self.sid)
        self.assertEqual(s["name"], "raspberry-pi")
        self.assertEqual(s["default_name"], "raspberry-pi")

    def test_rename_then_heartbeat_keeps_the_name(self):
        satellites.heartbeat(self.sid, "raspberry-pi")
        satellites.set_state(self.sid, name="Kitchen")
        self.assertEqual(satellites.get(self.sid)["name"], "Kitchen")
        # The device keeps checking in under its own name - the override must survive.
        satellites.heartbeat(self.sid, "raspberry-pi")
        s = satellites.get(self.sid)
        self.assertEqual(s["name"], "Kitchen")          # sticky
        self.assertEqual(s["default_name"], "raspberry-pi")

    def test_empty_name_reverts_to_device_name(self):
        satellites.heartbeat(self.sid, "raspberry-pi")
        satellites.set_state(self.sid, name="Kitchen")
        satellites.set_state(self.sid, name="   ")       # blank ⇒ revert
        self.assertEqual(satellites.get(self.sid)["name"], "raspberry-pi")

    def test_mute_and_volume_do_not_touch_the_name(self):
        satellites.heartbeat(self.sid, "raspberry-pi")
        satellites.set_state(self.sid, name="Shop")
        satellites.set_state(self.sid, muted=True, volume=40)
        s = satellites.get(self.sid)
        self.assertEqual(s["name"], "Shop")
        self.assertTrue(s["muted"])
        self.assertEqual(s["volume"], 40)


if __name__ == "__main__":
    unittest.main()
