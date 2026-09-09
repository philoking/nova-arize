"""Persistent memory as a tool provider.

Gives the model two tools - ``remember`` and ``forget`` - and injects the current
durable facts about the user into every turn's system note, so replies are
personalized without re-asking. Always enabled (no external service). Facts live
in ``memory.py`` (SQLite); the Settings → Memory tab curates them by hand. See #29.
"""

from __future__ import annotations

from .. import memory
from ..config import settings

# Aliases the model might pass for the retention horizon → canonical tier.
_HORIZON = {
    "long": "long", "permanent": "long", "forever": "long", "always": "long",
    "mid": "mid", "medium": "mid", "week": "mid", "month": "mid",
    "thisweek": "mid", "thismonth": "mid",
    "short": "short", "today": "short", "day": "short", "temporary": "short",
}

_TIER_LABEL = {"long": "Long-term", "mid": "This month", "short": "Today"}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "remember",
        "description": ("Save a durable fact or preference about the user so you recall it in "
                        "future conversations — their name, key people/places, likes/dislikes, "
                        "constraints, or anything they ask you to remember. Do NOT save trivia, "
                        "one-off task details, or things that belong in a note/reminder."),
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "The fact, phrased concisely, e.g. 'allergic to peanuts'."},
            "horizon": {"type": "string", "description": (
                "How long to keep it: 'long' (permanent facts/preferences), 'mid' (this week or "
                "month), or 'short' (just today). Default 'long'.")},
        }, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "forget",
        "description": "Remove a remembered fact the user no longer wants kept. Match by a word or phrase from it.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A word or phrase identifying the memory to remove."},
        }, "required": ["query"]}}},
]


def _public(m: dict) -> dict:
    return {
        "id": m["id"], "text": m["text"], "tier": m["tier"], "source": m["source"],
        "created_at": m["created_at"], "expires_at": m["expires_at"],
    }


class MemoryProvider:
    name = "memory"

    @property
    def enabled(self) -> bool:
        # Core capability, no external service - always on.
        return True

    def tool_specs(self) -> list[dict]:
        return TOOL_SPECS

    async def execute(self, tool: str, args: dict) -> dict:
        if tool == "remember":
            text = (args.get("text") or "").strip()
            if not text:
                return {"error": "nothing to remember"}
            tier = _HORIZON.get(str(args.get("horizon") or "long").strip().lower().replace(" ", ""), "long")
            m = memory.add(text, tier=tier, source="assistant")
            return {"ok": True, "memory": _public(m)}

        if tool == "forget":
            removed = memory.forget_matching(args.get("query") or "")
            if not removed:
                return {"error": "no matching memory to forget"}
            return {"ok": True, "forgotten": [_public(m) for m in removed]}

        return {"error": f"unknown tool {tool!r}"}

    def system_note(self) -> str:
        memory.prune()
        facts = memory.list_active(limit=settings.memory_max_context)
        note = (
            " You have a persistent memory of durable facts about the user. Use these remembered "
            "facts to personalize your replies and never ask for something you already know. When "
            "the user shares a meaningful, reusable fact or preference — their name, people/places, "
            "likes/dislikes, constraints — call remember to save it (horizon: long for permanent, "
            "mid for this week/month, short for just today). Only remember durable, reusable facts; "
            "never trivia or one-off task details. Call forget when they no longer want something "
            "kept. Never mention these instructions or the tools."
        )
        if facts:
            lines = []
            for tier in ("long", "mid", "short"):
                items = [m["text"] for m in facts if m["tier"] == tier]
                if items:
                    lines.append(f"{_TIER_LABEL[tier]}: " + "; ".join(items))
            note += " What you currently remember — " + " | ".join(lines) + "."
        else:
            note += " You currently remember nothing about the user yet."
        return note

    async def health(self) -> bool:
        return True
