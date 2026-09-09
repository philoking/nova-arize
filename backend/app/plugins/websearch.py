"""Web search as a tool provider.

Thin adapter over ``services/searxng.py``, presented to the registry under the
``web`` namespace so the tool reads as ``web__web_search``. Enabled only when a
SearXNG URL is configured.
"""

from __future__ import annotations

from ..config import settings
from ..services import searxng

_SYSTEM_NOTE = (
    " You have web_search for looking things up on the public internet, so never "
    "tell the user you lack internet access. Answer general knowledge you already "
    "know directly and confidently. When the user asks about something current, "
    "external, or that you are not sure of — news, weather, sports, prices, recent "
    "events, specific facts — call web_search and answer from the results in your "
    "own words. Prefer one focused query."
)


class WebSearchProvider:
    name = "web"

    @property
    def enabled(self) -> bool:
        return settings.websearch_enabled

    def tool_specs(self) -> list[dict]:
        return searxng.TOOL_SPECS

    async def execute(self, tool: str, args: dict) -> dict:
        return await searxng.execute_tool(tool, args)

    def system_note(self) -> str:
        return _SYSTEM_NOTE

    async def health(self) -> bool:
        return await searxng.health()
