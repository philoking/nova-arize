"""Unit tests for activity-log noise classification (#55).

Covers the pure classifier that decides which log lines are routine polling
(hideable) vs. meaningful activity. No DB/network.

Stdlib ``unittest``. Run from ``backend/``:  python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest

os.environ["NOVA_VOICE_HISTORY_DB"] = os.path.join(tempfile.mkdtemp(), "test_logstore.db")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import logstore  # noqa: E402


def access(path):
    return f'198.51.100.9:1234 - "GET {path} HTTP/1.1" 200'


class NoiseClassifier(unittest.TestCase):
    def n(self, logger, msg):
        return logstore._is_polling_noise(logger, msg)

    def test_polling_paths_are_noise(self):
        for path in ("/api/alarms/ringing", "/api/timers/fired?since=1", "/api/timers",
                     "/api/satellites/abc/heartbeat", "/api/config", "/api/log?limit=500",
                     "/api/health"):
            self.assertTrue(self.n("uvicorn.access", access(path)), path)

    def test_meaningful_requests_are_not_noise(self):
        for path in ("/api/chat", "/api/stt", "/api/tts", "/api/escalate", "/api/memories"):
            self.assertFalse(self.n("uvicorn.access", access(path)), path)

    def test_upstream_voice_fetch_is_noise_but_speech_is_not(self):
        self.assertTrue(self.n("httpx", "HTTP Request: GET http://nova:8880/v1/audio/voices \"200 OK\""))
        self.assertFalse(self.n("httpx", "HTTP Request: POST http://nova:8880/v1/audio/speech \"200 OK\""))

    def test_our_own_events_are_not_noise(self):
        self.assertFalse(self.n("nova", "chat turn [satellite/voice] 'what time is it' -> skills=none"))
        self.assertFalse(self.n("nova", "tool call get_state(...)"))


if __name__ == "__main__":
    unittest.main()
