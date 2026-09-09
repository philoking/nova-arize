"""Timers & reminders as a tool provider.

Lets the model start countdown timers and time-based reminders from either
surface. Times are resolved to a concrete epoch here: ``start_timer`` takes a
duration, ``set_reminder`` takes either a relative ``in_seconds`` or an absolute
ISO-8601 local ``fire_at``. The current local time is injected into the system
note so the model can do the arithmetic, and each created timer is tagged with
the surface the request came from (see ``timers.current_source``).
"""

from __future__ import annotations

from datetime import datetime

from .. import timers
from ..config import settings

_DAY_NAMES = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
              "fri": "Friday", "sat": "Saturday", "sun": "Sunday"}
# Full day names and shorthands the model might pass for a weekly recurrence.
_DAY_ALIASES = {
    "monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu",
    "friday": "fri", "saturday": "sat", "sunday": "sun",
}


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _normalize_repeat(raw) -> str | None:
    """Coerce a model-supplied repeat value into a canonical recurrence token, or
    None if it isn't a recognised recurrence."""
    if not raw:
        return None
    s = str(raw).strip().lower().replace(" ", "")
    if s in ("daily", "everyday", "every-day"):
        return "daily"
    if s in ("weekdays", "weekday", "everyweekday"):
        return "weekdays"
    if s in ("weekends", "weekend"):
        return "weekends"
    if s in timers._DOW:
        return f"weekly:{s}"
    if s in _DAY_ALIASES:
        return f"weekly:{_DAY_ALIASES[s]}"
    if s.startswith("weekly:"):
        dow = s.split(":", 1)[1]
        dow = _DAY_ALIASES.get(dow, dow)[:3]
        return f"weekly:{dow}" if dow in timers._DOW else None
    if s.startswith("monthly:"):
        try:
            day = int(s.split(":", 1)[1])
        except ValueError:
            return None
        return f"monthly:{day}" if 1 <= day <= 31 else None
    return None


def _when(epoch: int) -> str:
    # 12-hour time without a leading zero, built portably (strftime's "%-I" is a
    # glibc-only extension that raises on Windows).
    dt = datetime.fromtimestamp(epoch, timers._tz())
    hour = dt.strftime("%I").lstrip("0") or "12"
    return f"{hour}:{dt.strftime('%M %p')}".strip()


def format_timer(t: dict) -> dict:
    """Shape a raw timer row for a tool result / the UI, with a spoken-friendly
    time, cadence label, and alarm flag. Shared by the provider and /api/timers."""
    recurrence = t.get("recurrence")
    return {
        "id": t["id"], "kind": t["kind"], "label": t["label"],
        "fire_at": t["fire_at"], "source": t["source"], "status": t["status"],
        "in_seconds": max(0, t["fire_at"] - timers._now()),
        "when": _when(t["fire_at"]),
        "recurrence": recurrence,
        "cadence": _cadence_text(recurrence),
        "is_alarm": bool(t.get("is_alarm")),
    }


def _cadence_text(recurrence: str | None) -> str | None:
    """A short human label for the recurrence, for the panel and spoken confirmation."""
    if not recurrence:
        return None
    if recurrence in ("daily", "weekdays", "weekends"):
        return recurrence.capitalize()
    if recurrence.startswith("weekly:"):
        return f"Every {_DAY_NAMES.get(recurrence.split(':', 1)[1], recurrence)}"
    if recurrence.startswith("monthly:"):
        try:
            return f"Monthly on the {_ordinal(int(recurrence.split(':', 1)[1]))}"
        except ValueError:
            return recurrence
    return recurrence

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "start_timer",
        "description": "Start a countdown timer. Use for 'set a timer for N minutes'.",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "Short name for what it's for, e.g. 'pasta'."},
            "duration_seconds": {"type": "integer", "description": "Seconds until it goes off."},
        }, "required": ["duration_seconds"]}}},
    {"type": "function", "function": {
        "name": "set_reminder",
        "description": ("Set a reminder. For 'in N minutes/hours' pass in_seconds; for an absolute "
                        "time like '5pm' pass fire_at as an ISO-8601 local datetime. For a repeating "
                        "reminder ('every day at 7am', 'every weekday', 'every Monday') also pass repeat."),
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "What to be reminded about, e.g. 'call mom'."},
            "in_seconds": {"type": "integer", "description": "Seconds from now (relative reminders)."},
            "fire_at": {"type": "string", "description": "Absolute local time, ISO-8601 (e.g. 2026-07-08T17:00:00)."},
            "repeat": {"type": "string", "description": (
                "Optional recurrence: 'daily', 'weekdays', 'weekends', 'weekly:<dow>' "
                "(dow = mon/tue/wed/thu/fri/sat/sun), or 'monthly:<day>' (1-31). Include "
                "fire_at with the time of day for the first occurrence.")},
            "alarm": {"type": "boolean", "description": (
                "True for an alarm — it rings with a repeating chime until dismissed, "
                "rather than announcing once. Use for 'set an alarm for 7am'.")},
        }, "required": ["label"]}}},
    {"type": "function", "function": {
        "name": "list_timers",
        "description": "List the active timers and reminders.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "cancel_timer",
        "description": "Cancel an active (not-yet-fired) timer, reminder, or alarm by its label.",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "The label (or part of it) to cancel."},
        }, "required": ["label"]}}},
    {"type": "function", "function": {
        "name": "dismiss_alarm",
        "description": ("Stop a currently ringing alarm. Use for 'stop', 'dismiss', "
                        "'turn off the alarm'. Omit label to stop all ringing alarms."),
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "Optional label (or part of it) of the alarm to stop."},
        }}}},
]


class TimersProvider:
    name = "timers"

    @property
    def enabled(self) -> bool:
        # Core capability, no external service to configure - always on.
        return True

    def _tz(self):
        return timers._tz()

    def _public(self, t: dict) -> dict:
        return format_timer(t)

    def tool_specs(self) -> list[dict]:
        return TOOL_SPECS

    def _resolve_when(self, args: dict) -> int | None:
        if args.get("in_seconds") is not None:
            try:
                return timers._now() + int(args["in_seconds"])
            except (TypeError, ValueError):
                return None
        raw = args.get("fire_at")
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(str(raw))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._tz())
        return int(dt.timestamp())

    async def execute(self, tool: str, args: dict) -> dict:
        source = timers.current_source.get()
        if tool == "start_timer":
            try:
                secs = int(args.get("duration_seconds") or 0)
            except (TypeError, ValueError):
                return {"error": "duration_seconds must be a whole number of seconds"}
            if secs <= 0:
                return {"error": "duration_seconds must be positive"}
            t = timers.create("timer", args.get("label") or "timer", timers._now() + secs, source)
            return {"ok": True, "timer": self._public(t)}

        if tool == "set_reminder":
            recurrence = None
            if args.get("repeat"):
                recurrence = _normalize_repeat(args["repeat"])
                if recurrence is None:
                    return {"error": "unknown repeat; use daily, weekdays, weekends, "
                                     "weekly:<dow>, or monthly:<day>"}
            fire_at = self._resolve_when(args)
            if fire_at is None:
                return {"error": "provide either in_seconds or a valid ISO-8601 fire_at"}
            # For a repeating reminder whose first time-of-day has already passed,
            # roll forward to the next occurrence instead of rejecting it.
            if recurrence:
                while fire_at <= timers._now():
                    nxt = timers.next_occurrence(recurrence, fire_at)
                    if nxt is None:
                        break
                    fire_at = nxt
            if fire_at <= timers._now():
                return {"error": "that time is in the past"}
            is_alarm = bool(args.get("alarm"))
            t = timers.create("reminder", args.get("label") or "reminder", fire_at, source,
                              recurrence=recurrence, is_alarm=is_alarm)
            return {"ok": True, "reminder": self._public(t)}

        if tool == "list_timers":
            return {"timers": [self._public(t) for t in timers.list_active()]}

        if tool == "cancel_timer":
            cancelled = timers.cancel_by_label(args.get("label") or "")
            if not cancelled:
                return {"error": "no matching active timer or reminder"}
            return {"ok": True, "cancelled": self._public(cancelled)}

        if tool == "dismiss_alarm":
            dismissed = timers.acknowledge_ringing(args.get("label"))
            if not dismissed:
                return {"error": "no alarm is currently ringing"}
            return {"ok": True, "dismissed": [self._public(d) for d in dismissed]}

        return {"error": f"unknown tool {tool!r}"}

    def system_note(self) -> str:
        now = datetime.now(self._tz())
        hour = now.strftime("%I").lstrip("0") or "12"
        stamp = f"{now.strftime('%A %Y-%m-%d')} {hour}:{now.strftime('%M %p %Z')}".strip()
        note = (
            f" You can set countdown timers and reminders with the provided tools. The current "
            f"local time is {stamp}. For 'in N minutes/hours' use "
            f"start_timer (duration_seconds) or set_reminder (in_seconds); for an absolute time like "
            f"'5pm' use set_reminder with fire_at as an ISO-8601 local datetime. For a repeating "
            f"reminder ('every day at 7am', 'every weekday', 'every Monday', 'monthly on the 1st') pass "
            f"repeat (daily / weekdays / weekends / weekly:<dow> / monthly:<day>) together with a fire_at "
            f"for the first time. For an alarm ('set an alarm for 7am') that rings until dismissed, pass "
            f"alarm=true. Always include a short label. After setting one, confirm the time (and cadence) "
            f"in words. Use list_timers and cancel_timer to manage them, and dismiss_alarm to stop a "
            f"ringing alarm ('stop', 'dismiss')."
        )
        # When an alarm is actually ringing, make "stop" unambiguous: the user is
        # almost certainly trying to silence it, so bias hard toward dismiss_alarm.
        ringing = timers.ringing()
        if ringing:
            labels = ", ".join(dict.fromkeys(r["label"] for r in ringing))
            note += (
                f" AN ALARM IS RINGING RIGHT NOW ({labels}). If the user says anything like 'stop', "
                f"'dismiss', 'turn it off', 'quiet', 'enough', 'okay', or otherwise wants it to stop, "
                f"call dismiss_alarm immediately (no label needed) and do nothing else first."
            )
        return note

    async def health(self) -> bool:
        return True
