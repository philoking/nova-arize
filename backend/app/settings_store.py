"""Persisted, user-editable runtime settings.

The env vars in ``config.py`` are the *defaults*; this store holds the small set
of values the user can change at runtime from the web UI's Settings screen
(currently just the TTS voice). Overrides are persisted to a JSON file so they
survive restarts and - crucially - apply to *every* surface that hits this
backend, including the Pi satellite (which sends no voice of its own, so it
inherits whatever the server default is on its next turn).

To make another value user-editable, add it to ``_DEFAULTS`` with a lambda that
reads its default from ``settings``. The API and the store need no other change.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading

from .config import settings

# Keys the UI may write, each mapped to the env-derived default it falls back to.
# This whitelist is the only thing gating what the Settings API will persist.
# Persona prompts resolve to the built-in default via prompt_for when unset, so the
# Settings screen shows the real prompt and "Reset" just clears the override.
_DEFAULTS = {
    "voice": lambda: settings.voice,
    "model": lambda: settings.model,
    "voice_prompt": lambda: settings.prompt_for("voice"),
    "text_prompt": lambda: settings.prompt_for("text"),
    # Auto-escalate hard turns to Claude ("1"/"0"). Only takes effect when an
    # Anthropic key is configured (claude_enabled); the chat route gates on both.
    "auto_escalate": lambda: "1" if settings.auto_escalate else "0",
    # Phone notifications over MQTT (#16). Stored as strings like everything here:
    # the toggle as "1"/"0", categories as a comma-separated list.
    "notifications_enabled": lambda: "1" if settings.notifications_enabled else "0",
    "notification_categories": lambda: settings.notification_categories,
    "mqtt_address": lambda: settings.mqtt_address,
    "mqtt_topic": lambda: settings.mqtt_topic,
}


class SettingsStore:
    """A thin, file-backed overlay of user overrides on top of the env config."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._overrides: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        # Ignore anything not (or no longer) writable so a stale file can't
        # resurrect a retired key.
        return {k: str(v) for k, v in data.items() if k in _DEFAULTS and v is not None}

    @staticmethod
    def writable_keys() -> list[str]:
        return list(_DEFAULTS)

    def get(self, key: str) -> str:
        """The effective value: the saved override, else the env default."""
        if key in self._overrides:
            return self._overrides[key]
        return _DEFAULTS[key]()

    def all(self) -> dict[str, str]:
        """Every writable setting's effective value."""
        return {k: self.get(k) for k in _DEFAULTS}

    def set(self, key: str, value: str) -> None:
        """Persist an override. Raises ``KeyError`` for a non-writable key."""
        if key not in _DEFAULTS:
            raise KeyError(key)
        with self._lock:
            self._overrides[key] = value
            self._write()

    def reset(self, key: str) -> None:
        """Drop the override so the key falls back to its default. No-op if unset."""
        if key not in _DEFAULTS:
            raise KeyError(key)
        with self._lock:
            if key in self._overrides:
                del self._overrides[key]
                self._write()

    def _write(self) -> None:
        """Atomically replace the file so a crash mid-write can't corrupt it."""
        directory = os.path.dirname(self._path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._overrides, f, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


store = SettingsStore(settings.settings_file)
