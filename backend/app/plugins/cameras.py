"""Cameras as a tool provider (#68).

Exposes one tool, ``show_camera``, that puts a Frigate camera's live feed on
screen. The model picks a camera by its exact id from the list injected into the
system note (mirrors how Home Assistant injects its device allowlist); the tool
validates it and confirms, and the streamed ``{"tool": "show_camera", …}`` event
is what actually opens the modal in the PWA. Enabled only when Frigate is
configured (``NOVA_FRIGATE_URL``).
"""

from __future__ import annotations

from ..config import settings
from ..services import frigate

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "show_camera",
            "description": (
                "Put a security camera's live video feed on the user's screen. Use this whenever "
                "they ask to see, pull up, check, or watch a camera or a place a camera covers "
                "(e.g. 'show me the front porch', 'pull up the driveway', 'let me see the chicken "
                "coop'). Pass the exact camera id from the list of cameras in your system note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera": {
                        "type": "string",
                        "description": "A camera id from the list in the system note, e.g. 'front_porch_camera'.",
                    },
                },
                "required": ["camera"],
            },
        },
    },
]


class CamerasProvider:
    name = "cameras"

    @property
    def enabled(self) -> bool:
        return settings.frigate_enabled

    def tool_specs(self) -> list[dict]:
        return TOOL_SPECS

    async def execute(self, tool: str, args: dict) -> dict:
        if tool != "show_camera":
            return {"error": f"unknown tool: {tool}"}
        return await frigate.show_camera(args.get("camera", ""))

    def system_note(self) -> str:
        cams = frigate.cameras()
        if not cams:
            return (" You have a show_camera tool but no cameras are currently available, so tell the "
                    "user you can't pull up a camera right now if they ask.")
        listing = "; ".join(f"{c['name']} (id: {c['id']})" for c in cams)
        return (" To put a live camera feed on screen you MUST call show_camera(camera=<id>) — that tool "
                "call is the only thing that opens the feed. When the user asks to see/pull up/check/watch "
                "a camera or a place one covers, ALWAYS call show_camera with the exact id of the best-"
                "matching camera BEFORE replying; if none clearly matches, ask which instead. Never say "
                "you're showing a camera (e.g. 'here's the front porch') unless you actually called "
                "show_camera this turn — saying it without the call shows the user nothing. After the "
                f"call, confirm in one short sentence. Available cameras — {listing}.")

    async def health(self) -> bool:
        return await frigate.health()
