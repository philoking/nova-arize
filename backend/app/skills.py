"""Capability routing - narrow the tool surface the model sees per turn (#35/#39).

As capabilities grow, handing the model every tool (dozens) degrades a small
model's tool-selection: it skips the obvious call and free-associates. So instead
of one flat list, tools are grouped into **skills** - small, intent-scoped bundles
- and a **router** picks the 1-2 skills a turn needs. The agent loop then sees only
those skills' tools (typically 2-6, never all of them) and only their prompt
fragments, which keeps selection reliable as the catalog keeps growing.

Routing is **deterministic**: a keyword/intent match selects the relevant skills
with no model call, so it adds zero latency (important on a shared GPU). If nothing
matches, the turn runs tool-free (plain conversation) and the model answers from
its own knowledge. When a tool-needing phrasing isn't caught, broaden that skill's
``keywords`` here.

This module is pure/deterministic data + helpers; the Registry (which owns the
provider instances and the merged tool specs) drives the routing - see
``plugins/registry.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from . import memory, timers
from .config import settings


@dataclass(frozen=True)
class Skill:
    #: Stable id, used by the router and in tool grouping.
    name: str
    #: One line shown to the LLM router so it can classify against it.
    description: str
    #: Bare-word/phrase triggers for the deterministic fast-path (lowercased,
    #: punctuation-free; matched as whole words/phrases against the message).
    keywords: tuple[str, ...]
    #: Namespaced tool names (provider__tool) this skill exposes.
    tools: tuple[str, ...]
    #: Gated prompt fragment - added only when this skill is selected.
    note: str = ""
    #: If set, use this provider's live system_note() instead of note
    #: (for skills whose guidance is dynamic, e.g. Home Assistant's device list).
    provider_note: str | None = None


# The catalog. Adding a capability = adding a Skill here (and its provider's
# tools); the model's per-turn tool surface stays small regardless.
SKILLS: tuple[Skill, ...] = (
    Skill(
        name="home",
        description="Control or check smart-home devices — lights, thermostat, locks, switches, plugs, fans.",
        keywords=("turn on", "turn off", "switch on", "switch off", "lights", "light", "lamp",
                  "thermostat", "temperature", "heat", "cool", "ac", "air conditioning", "lock",
                  "unlock", "locked", "unlocked", "locks", "plug", "outlet", "dim", "brighten",
                  "brightness", "fan", "set the"),
        tools=("home_assistant__find_entities", "home_assistant__get_state", "home_assistant__call_service"),
        provider_note="home_assistant",
    ),
    Skill(
        name="lists",
        description="Standing named checklists (shopping / grocery / to-do lists): add, read, check off, remove, or clear items.",
        keywords=("list", "lists", "shopping", "grocery", "groceries", "checklist", "cart",
                  "check off", "cross off", "tick off"),
        tools=("notes__add_to_list", "notes__read_list", "notes__create_list",
               "notes__check_item", "notes__remove_item", "notes__clear_list"),
        note=(" For a standing named checklist (shopping, groceries, to-do): add_to_list to add an "
              "item, read_list to read it back, check_item to tick one off, remove_item to delete an "
              "item (if more than one matches, ask which), and clear_list to empty it. create_list "
              "only starts an empty one."),
    ),
    Skill(
        name="tasks",
        description="The user's to-do tasks in their daily notes: list open tasks, or add a task.",
        keywords=("task", "tasks", "to do", "to dos", "todo", "todos", "to-do"),
        tools=("notes__list_tasks", "notes__capture"),
        note=(" For ANY question about the user's tasks/to-dos ('do I have tasks', 'what's on my "
              "to-do list', 'any open tasks from yesterday'), call list_tasks with "
              "when='today'|'yesterday'|'recent'|'YYYY-MM-DD' — do NOT search for tasks. To add a "
              "task, use capture with section='tasks'."),
    ),
    Skill(
        name="notes",
        description="Read, search, or write the user's Obsidian notes — jot things down, look things up, or things they track (shows, episodes, books).",
        keywords=("note", "notes", "jot", "write down", "make a note", "in my notes", "wrote down",
                  "episode", "watch next", "reading", "book"),
        tools=("notes__capture", "notes__search_notes", "notes__read_note",
               "notes__create_note", "notes__append_note"),
        note=(" capture jots a line into today's daily note (Notes section by default). To answer any "
              "question about the user's notes — including shows/episodes/books they track — call "
              "search_notes first, then read_note on the best match before answering; never invent "
              "note contents. Use create_note/append_note for a standalone note."),
    ),
    Skill(
        name="projects",
        description="Create and manage the user's projects (folders of documents under Projects).",
        keywords=("project", "projects"),
        tools=("notes__create_project", "notes__add_project_doc", "notes__list_projects"),
        note=(" To start a project use create_project (it makes the folder with a README), then "
              "add_project_doc once per document; list_projects lists them."),
    ),
    Skill(
        name="cameras",
        description="Show a security/surveillance camera's live video feed on screen ('pull up the front porch').",
        # Camera-specific triggers only; the place names ('driveway', 'porch', …) are
        # injected dynamically from Frigate (see registry._camera_keywords), so 'pull
        # up the driveway' routes without the broad verbs over-routing other turns.
        keywords=("camera", "cameras", "webcam", "cctv", "surveillance", "live feed",
                  "pull up", "who's at", "whos at", "doorbell", "front door"),
        tools=("cameras__show_camera",),
        # Dynamic note: the provider lists the actual cameras so the model picks an
        # exact id (mirrors the 'home' skill's device list).
        provider_note="cameras",
    ),
    Skill(
        name="timers",
        description="Timers, reminders, and alarms: set, list, or cancel them, or dismiss a ringing alarm.",
        keywords=("timer", "timers", "remind", "reminder", "reminders", "alarm", "alarms", "wake me",
                  "snooze", "dismiss", "stop", "countdown", "in a minute", "in an hour"),
        tools=("timers__start_timer", "timers__set_reminder", "timers__list_timers",
               "timers__cancel_timer", "timers__dismiss_alarm"),
        note=(" For 'in N minutes/hours' use start_timer (duration_seconds) or set_reminder "
              "(in_seconds); for an absolute time like '5pm' use set_reminder fire_at (ISO-8601 "
              "local). For a repeat pass repeat (daily/weekdays/weekends/weekly:<dow>/monthly:<day>) "
              "with a fire_at; for an alarm that rings until dismissed pass alarm=true. Always include "
              "a short label and confirm the time. dismiss_alarm stops a ringing alarm."),
    ),
    Skill(
        name="web",
        description="Search the public internet for current or external facts — news, weather, prices, sports, general lookups.",
        keywords=("search", "look up", "google", "weather", "news", "price", "how much", "score",
                  "who is", "who's", "latest", "current", "look it up"),
        tools=("web__web_search",),
        note=(" Use web_search for current or external facts and answer from the results in your own "
              "words; prefer one focused query. Never say you lack internet access."),
    ),
    Skill(
        name="memory",
        description="Save or remove a durable fact or preference about the user (remember / forget).",
        keywords=("remember", "forget", "keep in mind", "note that i", "for future", "don't forget"),
        tools=("memory__remember", "memory__forget"),
        note=(" Save durable facts/preferences with remember(text, horizon: long/mid/short) — only "
              "meaningful, reusable facts, never trivia or one-off task details. forget(query) removes "
              "one."),
    ),
    Skill(
        name="manuals",
        description=("Look something up in the user's own uploaded product manuals / documentation "
                     "for their tools, appliances, and equipment — how-tos, settings, specs, parts, "
                     "error codes, maintenance procedures."),
        keywords=("manual", "manuals", "documentation", "owner's manual", "user guide", "how do i",
                  "how to", "how can i", "spec", "specs", "specification", "error code", "part number",
                  "torque", "calibrate", "maintenance", "troubleshoot"),
        tools=("manuals__search_manuals",),
        # Dynamic note: the provider lists the actual manuals on file so the model
        # knows what it can answer from (mirrors the 'home' skill's device list).
        provider_note="manuals",
    ),
    Skill(
        name="math",
        description=("Arithmetic and date math — add/subtract/multiply/divide, percentages, powers, "
                     "roots, and 'how long since/until' durations between dates."),
        keywords=("calculate", "calculator", "compute", "plus", "minus", "multiply", "multiplied",
                  "divided", "divide", "percent", "percentage", "square root", "squared", "cubed",
                  "factorial", "sum of", "product of",
                  # date/duration questions → date_diff
                  "how long", "how many days", "how many weeks", "how many months", "how many years",
                  "days since", "how old", "how long ago", "how long until", "days until", "days ago"),
        tools=("calculator__calculate", "calculator__date_diff"),
        note=(" For ANY arithmetic, call calculate with the expression — never compute in your head "
              "(you make mistakes on multi-digit math): '15% of 240' -> '0.15*240', 'square root of "
              "144' -> 'sqrt(144)', 'twelve times seven' -> '12*7'. For ANY 'how long since/until', "
              "'how many days/weeks/months since', or 'how old' question, call date_diff(start, end) "
              "with ISO dates (end omitted = today) — never count days across months yourself. Use a "
              "date the user gave or one you remember about them. Then state the exact result."),
    ),
)

_BY_NAME = {s.name: s for s in SKILLS}
_TIER_LABEL = {"long": "Long-term", "mid": "This month", "short": "Today"}


def available(tool_names: set[str]) -> list[Skill]:
    """Skills with at least one of their tools currently enabled (so skills whose
    provider is off - e.g. Obsidian offline - aren't offered to the router)."""
    return [s for s in SKILLS if any(t in tool_names for t in s.tools)]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation to spaces, pad with spaces for whole-word
    phrase matching (' turn on ' won't match 'return online')."""
    out = "".join(c if c.isalnum() or c.isspace() else " " for c in (text or "").lower())
    return " " + " ".join(out.split()) + " "


# Cues that a message continues the previous request: a pronoun standing in for
# the last thing acted on, or a continuation word. Such a turn inherits its skill,
# so "turn them back off" after "turn on the living room lights" routes to home.
_FOLLOWUP_CUES = frozenset({
    "it", "them", "they", "that", "those", "these", "this", "him", "her", "he", "she",
    "again", "back", "instead", "undo", "also", "too", "still", "one",
})


def is_followup(message: str) -> bool:
    return bool(set(_normalize(message).split()) & _FOLLOWUP_CUES)


# Math arrives as operator words ("twelve times seven") or symbols ("12*7"), which
# keyword matching misses. Added additively, so a false positive only exposes an
# unused calculate tool rather than hiding another skill's.
_MATH_WORDS = re.compile(
    r"\b(plus|minus|times|multipl(?:y|ied|ies)|divide[ds]?|percent(?:age)?|"
    r"squared|cubed|sqrt|factorial|modulo)\b")
# Operator symbols that never appear in dates/times/paths (so 2026-07-11 or 07/11
# don't read as math). Subtraction/division come through as the words minus/divide.
_MATH_SYMBOLS = re.compile(r"[+*×÷%^]")


def looks_like_math(message: str) -> bool:
    """True if the message is an arithmetic request (operator word, 'square root',
    or a digit-bearing operator symbol)."""
    t = (message or "").lower()
    return bool(_MATH_WORDS.search(t) or "square root" in t
                or (any(c.isdigit() for c in t) and _MATH_SYMBOLS.search(t)))


def deterministic_match(message: str, skills: list[Skill], extra: dict[str, list[str]] | None = None) -> list[str]:
    """Skills whose keywords (plus any dynamic ``extra`` keywords, e.g. HA device
    names) appear as whole words/phrases in the message. Empty when nothing hits -
    the caller then falls back to the LLM router."""
    padded = _normalize(message)
    extra = extra or {}
    hit = []
    for s in skills:
        kws = list(s.keywords) + extra.get(s.name, [])
        if any(f" {kw} " in padded for kw in kws if kw):
            hit.append(s.name)
    return hit


def always_on_context() -> str:
    """Context injected every turn regardless of routing: the current local time,
    what the assistant remembers about the user, and (if one is ringing) the
    dismiss-the-alarm directive - all broadly relevant and cheap."""
    parts: list[str] = []

    # Honesty guardrail: on a turn with no relevant tool the model must not fake an
    # action. Prevents "the lights are now off" when it never called a tool (#26).
    parts.append(" Only claim you did something — turned a device on/off, set a timer or reminder, "
                 "saved a note or list item — if you actually used a tool for it this turn and it "
                 "succeeded. If you have no tool to do what's asked, say you couldn't rather than "
                 "pretending you did it.")

    now = datetime.now(timers._tz())
    hour = now.strftime("%I").lstrip("0") or "12"
    parts.append(f" The current local time is {now.strftime('%A %Y-%m-%d')} {hour}:{now.strftime('%M %p %Z')}.".rstrip())

    facts = memory.list_active(limit=settings.memory_max_context)
    if facts:
        groups = []
        for tier in ("long", "mid", "short"):
            items = [m["text"] for m in facts if m["tier"] == tier]
            if items:
                groups.append(f"{_TIER_LABEL[tier]}: " + "; ".join(items))
        parts.append(" What you remember about the user — " + " | ".join(groups) +
                     ". Use these to personalise replies; don't ask for what you already know.")

    ringing = timers.ringing()
    if ringing:
        labels = ", ".join(dict.fromkeys(r["label"] for r in ringing))
        parts.append(f" AN ALARM IS RINGING RIGHT NOW ({labels}). If the user says anything like "
                     f"'stop', 'dismiss', 'turn it off', 'quiet', or 'enough', call dismiss_alarm "
                     f"immediately and do nothing else first.")
    return "".join(parts)
