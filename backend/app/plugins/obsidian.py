"""Obsidian notes as a tool provider.

Thin adapter over ``services/obsidian.py`` (the Local REST API client), presented
to the registry under the ``notes`` namespace so tools read as ``notes__capture``,
``notes__search_notes``, etc.
"""

from __future__ import annotations

from ..config import settings
from ..services import obsidian

_SYSTEM_NOTE = (
    " You can read and write the user's Obsidian notes. For a quick 'jot this "
    "down', use capture — it files the line under the Notes section of today's "
    "daily note by default. Only set section='tasks' (which becomes a checkbox) "
    "or section='goals' when the user explicitly says task/to-do or goal; "
    "otherwise leave it as notes. For a standing named checklist (e.g. a shopping "
    "or grocery list), use add_to_list to add an item ('add milk to my shopping "
    "list') and read_list to read it back; create_list only starts an empty one. "
    "A plain 'add a task' with no named list is a daily-note task (capture "
    "section='tasks'), NOT a list. For ANY question about the user's tasks or "
    "to-dos — 'do I have any tasks', 'any open tasks from yesterday', 'what's on "
    "my to-do list' — call list_tasks with when='today' | 'yesterday' | 'recent' "
    "| a 'YYYY-MM-DD' date; do NOT use search_notes for tasks. When the user "
    "marks a checklist item "
    "done/watched/finished, call check_item with the note's path and the item "
    "text — never say you marked it complete without actually calling check_item. "
    "To take an item off a list entirely ('remove milk', 'delete eggs from "
    "shopping'), use remove_item — that deletes it, unlike check_item which only "
    "ticks it off; if remove_item reports more than one match, ask which one. Use "
    "clear_list to empty a list ('clear my shopping list', or checked_only for "
    "'remove the done items'). "
    "The user tracks personal things in these notes — shows and episodes to watch, "
    "books to read, things to buy or do — so questions that sound external, like "
    "'which episode should I watch next' or 'what's next on my list', are really "
    "notes questions: search_notes before answering, and never tell the user you "
    "lack access to that or that it's online-only — it's their own note. "
    "To answer any question about their notes, call "
    "search_notes first, then read_note on the best match before answering — "
    "never invent note contents. To start a project, use create_project (it makes "
    "the folder with a README), then add_project_doc once per document you want "
    "inside it — don't create a separate note named after the project."
)


class ObsidianProvider:
    name = "notes"

    @property
    def enabled(self) -> bool:
        return settings.obsidian_enabled

    def tool_specs(self) -> list[dict]:
        return obsidian.TOOL_SPECS

    async def execute(self, tool: str, args: dict) -> dict:
        return await obsidian.execute_tool(tool, args)

    def system_note(self) -> str:
        return _SYSTEM_NOTE

    async def health(self) -> bool:
        return await obsidian.health()
