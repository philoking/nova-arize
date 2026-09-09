"""Unit tests for the cameras capability (#68).

Covers the pure Frigate helpers (friendly names, go2rtc stream mapping, spoken-
name resolution), the show_camera tool, and that camera names route the 'cameras'
skill. Stdlib ``unittest`` - the camera catalog is seeded directly, so no network.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import skills  # noqa: E402
from app.plugins.cameras import CamerasProvider  # noqa: E402
from app.services import frigate  # noqa: E402

ALL = list(skills.SKILLS)

# A trimmed Frigate /api/config-style cameras map: some declare their live stream
# explicitly, one omits it (so the <base>_main fallback is exercised).
FAKE_CAMERAS = {
    "front_porch_camera": {"live": {"streams": {"Main Stream": "front_porch_main"}}},
    "chicken_coop_inside": {"live": {"streams": {"Main": "chicken_coop_inside_main"}}},
    "driveway_entrance_camera": {},  # no live.streams -> fallback
}


def seed():
    """Populate the module cache the way refresh_cameras() would, without a fetch."""
    frigate._cameras = sorted(
        ({"id": cid, "name": frigate._friendly(cid), "stream": frigate._stream_for(cid, cfg)}
         for cid, cfg in FAKE_CAMERAS.items()),
        key=lambda c: c["name"],
    )


class FriendlyNames(unittest.TestCase):
    def test_strips_camera_suffix_and_titlecases(self):
        self.assertEqual(frigate._friendly("front_porch_camera"), "Front Porch")
        self.assertEqual(frigate._friendly("chicken_coop_inside"), "Chicken Coop Inside")
        self.assertEqual(frigate._friendly("house_garage_main_camera"), "House Garage")


class StreamMapping(unittest.TestCase):
    def test_uses_declared_live_stream(self):
        self.assertEqual(
            frigate._stream_for("front_porch_camera", FAKE_CAMERAS["front_porch_camera"]),
            "front_porch_main",
        )

    def test_falls_back_to_base_main(self):
        self.assertEqual(frigate._stream_for("driveway_entrance_camera", {}), "driveway_entrance_main")
        # A camera id without the _camera suffix still gets _main appended.
        self.assertEqual(frigate._stream_for("chicken_coop_inside", {}), "chicken_coop_inside_main")


class Resolve(unittest.TestCase):
    def setUp(self):
        seed()

    def test_exact_id(self):
        self.assertEqual(frigate.resolve("front_porch_camera")["id"], "front_porch_camera")

    def test_spoken_name(self):
        self.assertEqual(frigate.resolve("front porch")["id"], "front_porch_camera")
        self.assertEqual(frigate.resolve("show me the driveway")["id"], "driveway_entrance_camera")
        self.assertEqual(frigate.resolve("chicken coop")["id"], "chicken_coop_inside")

    def test_no_overlap_returns_none(self):
        self.assertIsNone(frigate.resolve("spaceship hangar"))
        self.assertIsNone(frigate.resolve(""))

    def test_filler_only_query_returns_none(self):
        # All stopwords (no discriminating token) must not match a random camera.
        self.assertIsNone(frigate.resolve("show me the camera please"))


class ShowCameraTool(unittest.TestCase):
    def setUp(self):
        seed()
        # Freeze the cache so ensure_fresh() doesn't try to re-fetch over the network.
        frigate._fetched_at = frigate.time.monotonic()

    def run_tool(self, camera):
        return asyncio.run(frigate.show_camera(camera))

    def test_resolves_and_echoes_the_id(self):
        out = self.run_tool("front porch")
        self.assertTrue(out["ok"])
        self.assertEqual(out["camera"], "front_porch_camera")
        self.assertEqual(out["name"], "Front Porch")

    def test_unknown_camera_lists_options(self):
        out = self.run_tool("spaceship")
        self.assertIn("error", out)
        self.assertIn("Front Porch", out["available"])


class Provider(unittest.TestCase):
    def setUp(self):
        seed()

    def test_system_note_lists_cameras_with_ids(self):
        note = CamerasProvider().system_note()
        self.assertIn("front_porch_camera", note)
        self.assertIn("Front Porch", note)
        self.assertIn("show_camera", note)

    def test_execute_rejects_unknown_tool(self):
        out = asyncio.run(CamerasProvider().execute("nope", {}))
        self.assertIn("error", out)


class ShowConfirmation(unittest.TestCase):
    """The heuristic behind the safety net that opens the modal when qwen3 confirms
    a camera in words but forgets the show_camera tool call (#68)."""

    def test_matches_show_phrasings(self):
        for r in ["Here's the front porch camera.", "Here is the driveway.",
                  "Pulling up the shop now.", "Showing the chicken coop.",
                  "Take a look at the pond.", "There's the back patio."]:
            self.assertTrue(frigate.looks_like_show_confirmation(r), r)

    def test_ignores_non_show_replies(self):
        for r in ["I can't show that camera right now.", "Which camera do you mean?",
                  "The front porch camera appears to be offline.", ""]:
            self.assertFalse(frigate.looks_like_show_confirmation(r), r)


class FallbackResolution(unittest.TestCase):
    """The fallback resolves the camera the model *named* in its reply first, so an
    ambiguous request ('the shop camera') still opens the one it meant."""

    def setUp(self):
        # A couple of same-prefix cameras to make the request ambiguous.
        frigate._cameras = sorted(
            ({"id": cid, "name": frigate._friendly(cid),
              "stream": frigate._stream_for(cid, {})}
             for cid in ["shop_inside_main_camera", "shop_outside_camera", "front_porch_camera"]),
            key=lambda c: c["name"],
        )

    def test_reply_disambiguates_the_request(self):
        # Request names only "shop"; the reply pins "Shop Inside".
        reply = "Here's the Shop Inside camera."
        cam = frigate.resolve(reply) or frigate.resolve("show me the shop camera")
        self.assertEqual(cam["id"], "shop_inside_main_camera")


class Routing(unittest.TestCase):
    def m(self, msg, extra=None):
        return skills.deterministic_match(msg, ALL, extra)

    def test_routes_on_generic_camera_words(self):
        self.assertIn("cameras", self.m("pull up the driveway"))
        self.assertIn("cameras", self.m("show me the front porch camera"))
        self.assertIn("cameras", self.m("who's at the front door"))

    def test_routes_on_dynamic_place_name(self):
        # A camera named for a place routes even without the word 'camera'.
        self.assertIn("cameras", self.m("what's happening at the pond", {"cameras": ["pond"]}))


if __name__ == "__main__":
    unittest.main()
