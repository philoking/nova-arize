"""Unit tests for recurring-reminder scheduling (#15).

Stdlib ``unittest``. Run from ``backend/``:  python -m unittest discover -s tests

Uses a fixed timezone so weekday/time assertions are deterministic regardless of
the host's zone.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

# Point the store at a throwaway DB before importing the app, so the Alarms tests
# never touch a real nova.db (settings.history_db is read once, at import).
os.environ["NOVA_VOICE_HISTORY_DB"] = os.path.join(tempfile.mkdtemp(), "test_timers.db")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import timers  # noqa: E402
from app.plugins.timers import _normalize_repeat, _cadence_text  # noqa: E402

# A fixed-offset zone keeps the weekday/time assertions deterministic and avoids
# depending on the tzdata package being present in the test interpreter.
TZ = timezone(timedelta(hours=-5))
timers._tz = lambda: TZ  # scheduling resolves times in this zone for the tests


def epoch(y, mo, d, h, mi):
    return int(datetime(y, mo, d, h, mi, tzinfo=TZ).timestamp())


def local(ep):
    return datetime.fromtimestamp(ep, TZ)


class NextOccurrence(unittest.TestCase):
    def test_daily_advances_one_day_same_time(self):
        # Wed 2026-07-08 07:00 → Thu 2026-07-09 07:00
        nxt = local(timers.next_occurrence("daily", epoch(2026, 7, 8, 7, 0)))
        self.assertEqual((nxt.year, nxt.month, nxt.day, nxt.hour, nxt.minute), (2026, 7, 9, 7, 0))

    def test_weekdays_skips_the_weekend(self):
        # Fri 2026-07-10 → next weekday is Mon 2026-07-13
        nxt = local(timers.next_occurrence("weekdays", epoch(2026, 7, 10, 9, 0)))
        self.assertEqual((nxt.month, nxt.day), (7, 13))
        self.assertEqual(nxt.weekday(), 0)  # Monday

    def test_weekends_from_a_weekday(self):
        nxt = local(timers.next_occurrence("weekends", epoch(2026, 7, 8, 8, 0)))  # Wed
        self.assertIn(nxt.weekday(), (5, 6))
        self.assertEqual((nxt.month, nxt.day), (7, 11))  # Sat

    def test_weekly_specific_dow(self):
        # From Wed 2026-07-08, next Monday is 2026-07-13
        nxt = local(timers.next_occurrence("weekly:mon", epoch(2026, 7, 8, 7, 0)))
        self.assertEqual(nxt.weekday(), 0)
        self.assertEqual((nxt.month, nxt.day), (7, 13))

    def test_monthly_same_day_next_month(self):
        nxt = local(timers.next_occurrence("monthly:15", epoch(2026, 7, 15, 6, 0)))
        self.assertEqual((nxt.month, nxt.day, nxt.hour), (8, 15, 6))

    def test_monthly_day_clamped_to_month_length(self):
        # Jan 31 → Feb has no 31st, clamp to 28 (2026 is not a leap year)
        nxt = local(timers.next_occurrence("monthly:31", epoch(2026, 1, 31, 6, 0)))
        self.assertEqual((nxt.month, nxt.day), (2, 28))

    def test_one_shot_and_unknown_return_none(self):
        self.assertIsNone(timers.next_occurrence(None, epoch(2026, 7, 8, 7, 0)))
        self.assertIsNone(timers.next_occurrence("hourly", epoch(2026, 7, 8, 7, 0)))


class NormalizeRepeat(unittest.TestCase):
    def test_canonical_and_aliases(self):
        self.assertEqual(_normalize_repeat("daily"), "daily")
        self.assertEqual(_normalize_repeat("every day"), "daily")
        self.assertEqual(_normalize_repeat("weekday"), "weekdays")
        self.assertEqual(_normalize_repeat("Monday"), "weekly:mon")
        self.assertEqual(_normalize_repeat("weekly:tue"), "weekly:tue")
        self.assertEqual(_normalize_repeat("weekly:saturday"), "weekly:sat")
        self.assertEqual(_normalize_repeat("monthly:1"), "monthly:1")

    def test_rejects_garbage(self):
        self.assertIsNone(_normalize_repeat("yearly"))
        self.assertIsNone(_normalize_repeat("monthly:41"))
        self.assertIsNone(_normalize_repeat(""))
        self.assertIsNone(_normalize_repeat(None))


class CadenceText(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(_cadence_text("daily"), "Daily")
        self.assertEqual(_cadence_text("weekdays"), "Weekdays")
        self.assertEqual(_cadence_text("weekly:mon"), "Every Monday")
        self.assertEqual(_cadence_text("monthly:1"), "Monthly on the 1st")
        self.assertEqual(_cadence_text("monthly:22"), "Monthly on the 22nd")
        self.assertIsNone(_cadence_text(None))


class Alarms(unittest.TestCase):
    """Alarm ringing/dismiss lifecycle against a throwaway on-disk DB."""

    def setUp(self):
        timers.init()
        timers._db().execute("DELETE FROM timers")  # isolate each test
        timers._db().commit()

    def _fire(self, tid):
        # Force the row due and run the scheduler's fire step.
        timers._db().execute("UPDATE timers SET fire_at=? WHERE id=?", (timers._now() - 1, tid))
        timers._db().commit()
        for t in timers.due(timers._now()):
            timers.mark_fired(t["id"])
            timers.rearm(t)

    def test_alarm_rings_then_dismisses(self):
        a = timers.create("reminder", "wake up", timers._now() + 1, "web", is_alarm=True)
        self._fire(a["id"])
        self.assertEqual([r["id"] for r in timers.ringing()], [a["id"]])
        # Alarms are kept out of the one-shot announce feed.
        self.assertEqual(timers.fired_since(0), [])
        self.assertTrue(timers.acknowledge(a["id"]))
        self.assertEqual(timers.ringing(), [])
        self.assertFalse(timers.acknowledge(a["id"]))  # already dismissed

    def test_non_alarm_uses_fired_feed_not_ringing(self):
        r = timers.create("reminder", "call mom", timers._now() + 1, "web")
        self._fire(r["id"])
        self.assertEqual(timers.ringing(), [])
        self.assertEqual([x["id"] for x in timers.fired_since(0)], [r["id"]])

    def test_recurring_alarm_rearms_next_as_alarm(self):
        a = timers.create("reminder", "meds", timers._now() + 1, "web", recurrence="daily", is_alarm=True)
        self._fire(a["id"])
        active = timers.list_active()
        self.assertEqual(len(active), 1)               # exactly the next occurrence
        self.assertTrue(active[0]["is_alarm"])          # still an alarm
        self.assertEqual(active[0]["recurrence"], "daily")
        self.assertEqual([r["id"] for r in timers.ringing()], [a["id"]])  # this one rings

    def test_acknowledge_ringing_by_label(self):
        a = timers.create("reminder", "gym", timers._now() + 1, "web", is_alarm=True)
        b = timers.create("reminder", "work", timers._now() + 1, "web", is_alarm=True)
        self._fire(a["id"]); self._fire(b["id"])
        dismissed = timers.acknowledge_ringing("gym")
        self.assertEqual([d["id"] for d in dismissed], [a["id"]])
        self.assertEqual([r["id"] for r in timers.ringing()], [b["id"]])


if __name__ == "__main__":
    unittest.main()
