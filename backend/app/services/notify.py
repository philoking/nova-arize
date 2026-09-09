"""Outbound phone notifications over MQTT (#16).

When a reminder/timer/alarm fires, publish a small JSON message to an MQTT topic
on the user's Mosquitto broker. A single Home Assistant automation subscribes to
that topic and delivers the push through the HA companion app - so reminders reach
the phone with the PWA closed, reusing the native push that HA already does well.

Design:
- **User-gated.** A global toggle plus a per-category allowlist (reminders / timers
  / alarms), all editable in Settings ▸ Notifications and persisted server-side.
- **Fire-and-forget.** Publishing is best-effort: a broker that's down, unset, or
  misconfigured logs a warning and returns False - it must never break a reply or
  kill the timer scheduler.
- **Decoupled.** nova only emits a generic notification *intent* to the bus; the
  single HA automation owns delivery policy (which device, formatting, whether to
  also announce on a speaker). Consistent with "HA is an actuator" (#3).

``aiomqtt`` is imported lazily so the gating helpers (and the app) load even when
the dependency isn't installed.
"""

from __future__ import annotations

import json
import logging

from ..config import settings
from ..settings_store import store

log = logging.getLogger(__name__)

# Categories the user can toggle, mapped to the spoken-notification title.
CATEGORIES = ("reminders", "timers", "alarms")
_TITLES = {"reminders": "Reminder", "timers": "Timer", "alarms": "Alarm"}


def enabled_categories() -> set[str]:
    """The categories the user has switched on (subset of ``CATEGORIES``)."""
    raw = store.get("notification_categories") or ""
    return {c.strip() for c in raw.split(",") if c.strip()}


def globally_enabled() -> bool:
    return store.get("notifications_enabled") == "1"


def is_enabled(category: str) -> bool:
    """True if a push should be sent for this category right now."""
    return globally_enabled() and category in enabled_categories()


def category_for(timer: dict) -> str:
    """Map a fired timer row to its notification category."""
    if timer.get("is_alarm"):
        return "alarms"
    return "timers" if timer.get("kind") == "timer" else "reminders"


async def _publish_raw(payload: dict) -> bool:
    """Connect, publish one JSON message, disconnect. Best-effort; never raises."""
    address = (store.get("mqtt_address") or "").strip()
    topic = (store.get("mqtt_topic") or "").strip()
    if not address or not topic:
        log.warning("notifications: mqtt_address/topic not set — nothing published")
        return False
    host, _, port_s = address.partition(":")
    port = int(port_s) if port_s.isdigit() else 1883
    try:
        import aiomqtt  # lazy: keep the module importable without the dependency

        auth = {}
        if settings.mqtt_username:
            auth = {"username": settings.mqtt_username, "password": settings.mqtt_password}
        async with aiomqtt.Client(hostname=host, port=port, **auth) as client:
            await client.publish(topic, payload=json.dumps(payload))
        log.info("notify → %s %s", topic, payload.get("message", ""))
        return True
    except Exception as exc:  # noqa: BLE001 - delivery must never break the caller
        log.error("notify publish failed (%s:%s): %s", host, port, exc)
        return False


async def publish(category: str, title: str, message: str, **extra) -> bool:
    """Publish a notification if the user has enabled this category. Returns sent?."""
    if not is_enabled(category):
        return False
    return await _publish_raw({"category": category, "title": title, "message": message, **extra})


async def emit_timer(timer: dict) -> bool:
    """Publish for a fired timer/reminder/alarm row (called from the scheduler)."""
    category = category_for(timer)
    label = timer.get("label") or "(no label)"
    return await publish(
        category,
        _TITLES.get(category, settings.assistant_name),
        label,
        id=timer.get("id"),
        source=timer.get("source"),
        ts=timer.get("fire_at"),
    )


async def send_test() -> bool:
    """Publish a fixed test message to the saved broker/topic, bypassing the
    enable/category gate so the pipe can be verified during setup. Still needs the
    address + topic to be set."""
    return await _publish_raw({
        "category": "test",
        "title": f"{settings.assistant_name} test",
        "message": "If you see this, notifications are working.",
    })
