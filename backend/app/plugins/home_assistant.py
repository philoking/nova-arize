"""Home Assistant as a tool provider.

The upstream HA client (search/read/control against HA's REST API) lives in
``services/home_assistant.py`` alongside the other service clients. This module
is the thin adapter that presents that client to the registry as a
``ToolProvider`` - it owns the HA-specific system note that used to live in the
LLM layer.
"""

from __future__ import annotations

from ..config import settings
from ..services import home_assistant as ha


class HomeAssistantProvider:
    name = "home_assistant"

    @property
    def enabled(self) -> bool:
        return settings.ha_enabled

    def tool_specs(self) -> list[dict]:
        return ha.TOOL_SPECS

    async def execute(self, tool: str, args: dict) -> dict:
        return await ha.execute_tool(tool, args)

    def system_note(self) -> str:
        """List the curated (labeled) devices so the model uses exact names/ids
        and is told, in no uncertain terms, not to touch anything else."""
        devices = ha.devices()
        if not devices:
            return (
                " No Home Assistant devices are set up for control yet. If the user asks to control or "
                f"read a device, explain that devices must be labeled '{settings.ha_label}' in Home "
                "Assistant before you can use them — don't attempt to control anything."
            )
        lines = "\n".join(f"- {d.get('name')} ({d.get('area') or 'no area'}) — {d.get('entity_id')}"
                          for d in devices)
        return (
            " You can control and read ONLY these Home Assistant devices:\n" + lines + "\n"
            "Match the user's request to exactly one device by its name and area, then use call_service "
            "to control it or get_state to read it, passing that exact entity_id. Never use or invent an "
            "entity_id that is not in this list. If the request doesn't clearly match one device (or "
            "matches none), ask the user which they mean instead of guessing. After acting, confirm briefly."
        )

    async def health(self) -> bool:
        return await ha.health()
