"""The link dashboard - a native, UI-editable replacement for the Homepage app.

Migrated off gethomepage.dev: instead of hand-editing YAML, the dashboard is a
small JSON document (groups → tiles) the user edits from Nova's own UI. The
packaged ``dashboard_seed.json`` is the *default* (imported one-time from the old
Homepage config so nothing was lost); once the user saves an edit, their version
is persisted to ``NOVA_VOICE_DASHBOARD_FILE`` on the ``/data`` volume and becomes
the source of truth. "Reset to default" is just deleting that file.

Shape (validated on every save so a bad client can't corrupt the file):

    {"version": 1, "groups": [
        {"id": str, "title": str, "tiles": [
            {"id": str, "name": str, "href": str, "icon": str, "desc": str,
             "metric": {"type": str}?}   # `metric` is a Phase-2 hint (Beszel/…),
        ]},                              # carried through but not yet rendered live
    ]}

A tile is just a link for now; the optional ``metric`` marks tiles that will grow
a live widget (Beszel/Portainer/Frigate) when those clients land.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading

from .config import settings

_SEED_PATH = os.path.join(os.path.dirname(__file__), "dashboard_seed.json")
_MAX_GROUPS = 40
_MAX_TILES = 200          # per group; generous for a homelab
_MAX_STR = 200            # cap any single string field
_ALLOWED_METRIC_TYPES = {"beszel", "portainer", "frigate"}


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or fallback


def _clean_str(v, limit: int = _MAX_STR) -> str:
    return str(v or "").strip()[:limit]


def _sanitize(config: dict) -> dict:
    """Coerce arbitrary client JSON into the strict dashboard shape. Never raises -
    drops/repairs anything malformed so a save can't wedge the file. Ids are filled
    in (and de-duplicated) so the UI always has stable keys."""
    groups_in = config.get("groups") if isinstance(config, dict) else None
    if not isinstance(groups_in, list):
        groups_in = []
    seen_gids: set[str] = set()
    groups = []
    for gi, g in enumerate(groups_in[:_MAX_GROUPS]):
        if not isinstance(g, dict):
            continue
        title = _clean_str(g.get("title"))
        if not title:
            continue
        gid = _slug(_clean_str(g.get("id")) or title, f"group-{gi}")
        while gid in seen_gids:
            gid += "-2"
        seen_gids.add(gid)
        tiles = []
        seen_tids: set[str] = set()
        for ti, t in enumerate(g.get("tiles", [])[:_MAX_TILES] if isinstance(g.get("tiles"), list) else []):
            if not isinstance(t, dict):
                continue
            name = _clean_str(t.get("name"))
            href = _clean_str(t.get("href"), 500)
            if not name and not href:
                continue
            tid = _slug(_clean_str(t.get("id")) or f"{gid}-{name}", f"{gid}-tile-{ti}")
            while tid in seen_tids:
                tid += "-2"
            seen_tids.add(tid)
            tile = {
                "id": tid,
                "name": name or href,
                "href": href,
                "icon": _clean_str(t.get("icon")),
                "desc": _clean_str(t.get("desc")),
            }
            m = t.get("metric")
            if isinstance(m, dict) and m.get("type") in _ALLOWED_METRIC_TYPES:
                tile["metric"] = {"type": m["type"]}
            tiles.append(tile)
        groups.append({"id": gid, "title": title, "tiles": tiles})
    return {"version": 1, "groups": groups}


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class DashboardStore:
    """File-backed dashboard document with a packaged seed as the default."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._seed = _sanitize(_read_json(_SEED_PATH) or {})

    def get(self) -> dict:
        """The current dashboard - the user's saved copy, else the packaged seed."""
        user = _read_json(self._path)
        return _sanitize(user) if user is not None else json.loads(json.dumps(self._seed))

    def save(self, config: dict) -> dict:
        """Validate + atomically persist the whole document. Returns the stored form."""
        clean = _sanitize(config)
        with self._lock:
            self._write(clean)
        return clean

    def reset(self) -> dict:
        """Drop the user's copy so the dashboard falls back to the packaged seed."""
        with self._lock:
            try:
                os.remove(self._path)
            except OSError:
                pass
        return self.get()

    def _write(self, data: dict) -> None:
        directory = os.path.dirname(self._path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


store = DashboardStore(settings.dashboard_file)
