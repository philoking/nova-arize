"""Obsidian notes client + tool definitions for the assistant's agent loop.

Talks to the "Local REST API & MCP Server" community plugin running in the
user's Obsidian on their Windows machine. Their vault is the source of truth -
we integrate with it rather than keeping a parallel store.

Exposed tools (bare names; the registry namespaces them under ``notes__``):

- ``capture`` - quick jot into today's daily note (the default target).
- ``create_note`` - create/overwrite a note at a path.
- ``append_note`` - append to a note (created if missing).
- ``read_note`` - read a note's contents.
- ``search_notes`` - full-text search across the vault.
- ``create_project`` - make a project folder + index note under Projects/.
- ``add_project_doc`` - add a document to a project folder.
- ``list_projects`` - list existing projects.

Reached over the LAN, so it only works while the Windows machine is awake and
the plugin is running; ``health`` reflects that. Every tool returns plain dicts
(errors as data) so the model can react rather than the loop crashing.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from ..config import settings

_MD = {"Content-Type": "text/markdown"}
# Characters Obsidian/most filesystems reject in a note or folder name. Spaces
# are fine in a vault, so we keep them.
_ILLEGAL = re.compile(r'[\\/:*?"<>|]+')


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.obsidian_url,
        headers={"Authorization": f"Bearer {settings.obsidian_api_key}"},
        timeout=15,
        verify=settings.obsidian_verify_ssl,
    )


def _safe_name(name: str) -> str:
    """A spoken name → a filesystem-safe note/folder name (keeps spaces)."""
    return _ILLEGAL.sub("", (name or "").strip()).strip(". ") or "untitled"


def _capitalize(s: str) -> str:
    """Capitalize just the first character, leaving the rest as typed."""
    s = (s or "").strip()
    return s[:1].upper() + s[1:] if s else s


def _md_path(name: str) -> str:
    """Ensure a note name ends in .md (leave an explicit path/extension alone)."""
    name = name.strip().lstrip("/")
    return name if name.lower().endswith(".md") else f"{name}.md"


def _vault(path: str) -> str:
    # Encode each segment but keep the slashes that separate folders.
    return "/vault/" + quote(path, safe="/")


def _project_dir(project: str) -> str:
    return f"{settings.obsidian_projects_dir.strip('/')}/{_safe_name(project)}"


def _list_name(name: str) -> str:
    """Normalise a spoken list name so 'shopping' and 'shopping list' (and
    'shopping checklist') all resolve to the same note - the model isn't
    consistent about whether it includes the trailing 'list' word."""
    n = re.sub(r"\s*(?:check)?list$", "", (name or "").strip(), flags=re.IGNORECASE)
    return _capitalize(_safe_name(n))


def _list_path(name: str) -> str:
    """Vault path for a named checklist, in the daily-notes folder (or root)."""
    folder = settings.obsidian_daily_dir.strip("/")
    safe = _list_name(name)
    return f"{folder}/{safe}.md" if folder else f"{safe}.md"


# Tools

# Sections in the daily-note template (## headings) and their spoken keys.
# Capture files a line under one of these headings rather than at the end of the
# note; a task becomes a checkbox to match the Tasks section's format.
DAILY_SECTIONS = {
    "notes": "Notes",
    "tasks": "Tasks",
    "goals": "Goals for Today",
}


def _daily_line(text: str, section: str) -> tuple[str, str, str] | tuple[None, None, None]:
    section = (section or "notes").strip().lower()
    heading = DAILY_SECTIONS.get(section)
    if heading is None:
        return None, None, None
    line = f"- [ ] {text}" if section == "tasks" else f"- {text}"
    return section, heading, line


async def capture(text: str, section: str = "notes") -> dict:
    """Add a line to today's daily note under a section heading.

    Defaults to the Notes section. ``section='tasks'`` files it under Tasks as a
    checkbox; ``section='goals'`` under Goals for Today. Inserts under the heading
    (via a PATCH) rather than appending to the end of the note.
    """
    text = (text or "").strip()
    if not text:
        return {"error": "nothing to capture"}
    sect, heading, line = _daily_line(text, section)
    if heading is None:
        return {"error": f"unknown section '{section}'; use notes, tasks, or goals"}
    async with _client() as c:
        resp = await c.patch(
            "/periodic/daily/",
            # Trim the section's existing trailing whitespace, then supply exactly
            # one newline before the line (so it sits tight under the previous
            # item - no blank gap between tasks) and a blank line after (to keep
            # separation from the next heading).
            content=f"\n{line}\n\n",
            headers={
                "Content-Type": "text/markdown",
                "Operation": "append",           # end of the section, before the next heading
                "Target-Type": "heading",
                "Target": heading,
                "Trim-Target-Whitespace": "true",
                "Create-Target-If-Missing": "true",
            },
        )
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"ok": True, "section": heading, "as_task": sect == "tasks"}


async def create_note(path: str, content: str = "") -> dict:
    """Create (or overwrite) a note at ``path`` (relative to the vault root)."""
    p = _md_path(path)
    async with _client() as c:
        resp = await c.put(_vault(p), content=content or "", headers=_MD)
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"ok": True, "path": p}


async def append_note(path: str, content: str) -> dict:
    """Append to an existing note (created empty first if it doesn't exist)."""
    p = _md_path(path)
    async with _client() as c:
        resp = await c.post(_vault(p), content=f"\n{content}\n", headers=_MD)
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"ok": True, "path": p}


async def read_note(path: str) -> dict:
    """Read a note's markdown contents."""
    p = _md_path(path)
    async with _client() as c:
        resp = await c.get(_vault(p), headers={"Accept": "text/markdown"})
        if resp.status_code == 404:
            return {"error": f"note not found: {p}"}
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"path": p, "content": resp.text}


async def search_notes(query: str, context_length: int = 100, limit: int = 10) -> dict:
    """Full-text search the vault; returns matching files with surrounding context."""
    if not (query or "").strip():
        return {"error": "empty search query"}
    async with _client() as c:
        resp = await c.post(
            "/search/simple/",
            params={"query": query, "contextLength": context_length},
        )
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
        results = resp.json() or []
    hits = [
        {
            "path": r.get("filename"),
            "score": r.get("score"),
            "context": " … ".join(m.get("context", "") for m in (r.get("matches") or [])[:3]),
        }
        for r in results[:limit]
    ]
    return {"count": len(results), "results": hits}


async def create_project(name: str, description: str = "") -> dict:
    """Create a project folder under Projects/, seeded with a README landing note.

    Obsidian's REST API can't make an empty folder, so we materialise it with a
    ``README.md`` (the project title as a heading, plus the optional description)
    rather than a note named after the project - which read as a confusing
    duplicate. Documents are added afterwards with add_project_doc.
    """
    safe = _safe_name(name)
    folder = _project_dir(name)
    desc = (description or "").strip()
    body = f"# {safe}\n\n{desc}\n" if desc else f"# {safe}\n\n"
    readme = f"{folder}/README.md"
    async with _client() as c:
        resp = await c.put(_vault(readme), content=body, headers=_MD)
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"ok": True, "project": safe, "readme": readme}


async def add_project_doc(project: str, title: str, content: str = "") -> dict:
    """Add a document to an existing project's folder."""
    path = f"{_project_dir(project)}/{_md_path(title)}"
    async with _client() as c:
        resp = await c.put(_vault(path), content=content or f"# {_safe_name(title)}\n\n", headers=_MD)
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"ok": True, "path": path}


async def list_projects() -> dict:
    """List the project folders under Projects/."""
    base = settings.obsidian_projects_dir.strip("/")
    async with _client() as c:
        resp = await c.get(f"/vault/{quote(base, safe='/')}/")
        if resp.status_code == 404:
            return {"projects": []}
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
        files = (resp.json() or {}).get("files", [])
    # Directory entries end with "/"; everything else is a loose file.
    projects = [f.rstrip("/") for f in files if f.endswith("/")]
    return {"count": len(projects), "projects": projects}


# Named checklists (shopping, to-do, …)
# A "list" is just a note in the daily-notes folder whose lines are checkboxes.
# Distinct from a one-off task, which goes into today's daily note via `capture`.

async def create_list(name: str) -> dict:
    """Start a named checklist note (e.g. 'shopping'). Idempotent: an existing
    list is left untouched, never wiped."""
    safe = _list_name(name)
    path = _list_path(name)
    async with _client() as c:
        existing = await c.get(_vault(path))
        if existing.status_code == 200:
            return {"ok": True, "list": safe, "path": path, "already_exists": True}
        resp = await c.put(_vault(path), content=f"# {safe}\n\n", headers=_MD)
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"ok": True, "list": safe, "path": path}


async def add_to_list(name: str, item: str) -> dict:
    """Add an item to a named checklist as an unchecked checkbox, creating the
    list (with a heading) if it doesn't exist yet."""
    safe = _list_name(name)
    item = _capitalize(item)
    if not item:
        return {"error": "nothing to add"}
    path = _list_path(name)
    async with _client() as c:
        cur = await c.get(_vault(path), headers={"Accept": "text/markdown"})
        if cur.status_code == 200:
            body = cur.text.rstrip("\n") + f"\n- [ ] {item}\n"
        elif cur.status_code == 404:
            body = f"# {safe}\n\n- [ ] {item}\n"
        else:
            return {"error": f"obsidian {cur.status_code}: {cur.text[:200]}"}
        resp = await c.put(_vault(path), content=body, headers=_MD)
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"ok": True, "list": safe, "path": path, "item": item}


async def read_list(name: str) -> dict:
    """Read a named checklist back."""
    safe = _list_name(name)
    path = _list_path(name)
    async with _client() as c:
        resp = await c.get(_vault(path), headers={"Accept": "text/markdown"})
        if resp.status_code == 404:
            return {"error": f"no list named '{safe}' yet"}
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"list": safe, "path": path, "content": resp.text}


# A Markdown task line: prefix ("- "/"* "), state (space or x), then the rest.
_CHECKBOX = re.compile(r"^(\s*[-*]\s+)\[([ xX])\](.*)$")


def _match_score(query: str, text: str) -> int:
    """Word-overlap score between a query and a line, with a substring bonus -
    used to pick which checklist line the user means."""
    qt = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    tt = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    score = len(qt & tt)
    if query and query.strip().lower() in (text or "").lower():
        score += 5
    return score


async def _resolve_note_path(c: httpx.AsyncClient, path_or_name: str) -> str | None:
    """Resolve a note the model referred to - a full vault path, a bare list name,
    or just the note's title - to a real path. The model often passes the friendly
    title, so we fall back to searching the vault and matching by filename."""
    # 1. As given (path, or a title → .md at the root).
    p = _md_path(path_or_name)
    if (await c.get(_vault(p))).status_code == 200:
        return p
    # 2. As a named list in the daily-notes folder.
    if "/" not in path_or_name:
        alt = _list_path(path_or_name)
        if (await c.get(_vault(alt))).status_code == 200:
            return alt
    # 3. Search the vault; match a result whose filename is (or contains) the name.
    base = re.sub(r"\.md$", "", path_or_name.rsplit("/", 1)[-1], flags=re.IGNORECASE).strip()
    if not base:
        return None
    resp = await c.post("/search/simple/", params={"query": base, "contextLength": 10})
    if resp.status_code != 200:
        return None
    target, fallback = base.lower(), None
    for r in (resp.json() or []):
        fn = r.get("filename", "")
        stem = re.sub(r"\.md$", "", fn.rsplit("/", 1)[-1], flags=re.IGNORECASE).lower()
        if stem == target:
            return fn
        if fallback is None and target in stem:
            fallback = fn
    return fallback


async def check_item(path: str, item: str, done: bool = True) -> dict:
    """Tick (or untick) a checklist item in a note.

    Finds the checkbox line best matching ``item`` that isn't already in the
    target state and flips ``[ ]`` ↔ ``[x]``. Returns an error (rather than a
    false success) when the note or the item can't be found, so the model can't
    claim it's done when it isn't.
    """
    async with _client() as c:
        p = await _resolve_note_path(c, path)
        if p is None:
            return {"error": f"couldn't find a note matching '{path}'"}
        r = await c.get(_vault(p), headers={"Accept": "text/markdown"})
        if r.status_code >= 400:
            return {"error": f"obsidian {r.status_code}: {r.text[:200]}"}

        lines = r.text.split("\n")
        best_i, best_score = -1, 0
        for i, line in enumerate(lines):
            m = _CHECKBOX.match(line)
            if not m or (m.group(2).lower() == "x") == done:
                continue  # not a task line, or already in the desired state
            score = _match_score(item, m.group(3))
            if score > best_score:
                best_i, best_score = i, score
        if best_i < 0:
            state = "unchecked" if done else "checked"
            return {"error": f"no {state} item matching '{item}' in {p}"}

        m = _CHECKBOX.match(lines[best_i])
        lines[best_i] = f"{m.group(1)}[{'x' if done else ' '}]{m.group(3)}"
        resp = await c.put(_vault(p), content="\n".join(lines), headers=_MD)
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"ok": True, "path": p, "item": m.group(3).strip(), "done": done}


async def remove_item(path: str, item: str) -> dict:
    """Delete a checklist item from a note - distinct from ticking it off.

    Removes the checkbox line best matching ``item``. If two items match equally
    well it returns the candidates instead of guessing, so the assistant can ask
    which one is meant. Errors honestly when nothing matches, rather than claiming
    a false success.
    """
    async with _client() as c:
        p = await _resolve_note_path(c, path)
        if p is None:
            return {"error": f"couldn't find a note matching '{path}'"}
        r = await c.get(_vault(p), headers={"Accept": "text/markdown"})
        if r.status_code >= 400:
            return {"error": f"obsidian {r.status_code}: {r.text[:200]}"}

        lines = r.text.split("\n")
        scored = []  # (score, index, text) for each checkbox line the query hits
        for i, line in enumerate(lines):
            m = _CHECKBOX.match(line)
            if not m:
                continue
            score = _match_score(item, m.group(3))
            if score > 0:
                scored.append((score, i, m.group(3).strip()))
        if not scored:
            return {"error": f"no item matching '{item}' in {p}"}
        top = max(s for s, _, _ in scored)
        best = [(i, text) for s, i, text in scored if s == top]
        if len(best) > 1:
            # Genuinely ambiguous - let the model ask which one.
            names = ", ".join(t for _, t in best)
            return {"error": "ambiguous", "candidates": [t for _, t in best],
                    "message": f"more than one item matches '{item}': {names}. Ask which one."}
        idx, text = best[0]
        del lines[idx]
        resp = await c.put(_vault(p), content="\n".join(lines), headers=_MD)
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"ok": True, "path": p, "removed": text}


async def clear_list(name: str, checked_only: bool = False) -> dict:
    """Remove items from a checklist - all of them, or only the checked ones
    (``checked_only=True``, for 'clear the done items'). Keeps the note + heading."""
    async with _client() as c:
        p = await _resolve_note_path(c, name)
        if p is None:
            return {"error": f"no list named '{name}' yet"}
        r = await c.get(_vault(p), headers={"Accept": "text/markdown"})
        if r.status_code >= 400:
            return {"error": f"obsidian {r.status_code}: {r.text[:200]}"}

        kept, removed = [], 0
        for line in r.text.split("\n"):
            m = _CHECKBOX.match(line)
            if m and (not checked_only or m.group(2).lower() == "x"):
                removed += 1
                continue
            kept.append(line)
        if removed == 0:
            return {"error": "no checked items to clear" if checked_only else f"'{name}' is already empty"}
        resp = await c.put(_vault(p), content="\n".join(kept), headers=_MD)
        if resp.status_code >= 400:
            return {"error": f"obsidian {resp.status_code}: {resp.text[:200]}"}
    return {"ok": True, "path": p, "removed": removed, "checked_only": checked_only}


# Tasks (checkboxes across daily notes)
# Tasks live as `- [ ]` lines in the daily notes (see `capture` section='tasks').
# Answering "do I have tasks?" needs a specific day's note, including past days,
# which `capture` (today-only) and `read_note` (exact path) can't do. `list_tasks`
# resolves a spoken scope to dates and reads each day's periodic endpoint.

def _tz():
    if settings.timezone:
        try:
            return ZoneInfo(settings.timezone)
        except Exception:  # noqa: BLE001 - a bad tz name falls back to local
            pass
    return datetime.now().astimezone().tzinfo


def _resolve_dates(when: str) -> list:
    """A spoken scope → the concrete date(s) to look in, newest first.
    'today' | 'yesterday' | 'recent' (last 7 days) | an ISO 'YYYY-MM-DD'."""
    today = datetime.now(_tz()).date()
    key = (when or "today").strip().lower()
    if key in ("", "today"):
        return [today]
    if key == "yesterday":
        return [today - timedelta(days=1)]
    if key in ("recent", "week", "this week", "lately"):
        return [today - timedelta(days=n) for n in range(7)]
    try:
        return [datetime.strptime(key, "%Y-%m-%d").date()]
    except ValueError:
        return [today]


async def _read_daily(c: httpx.AsyncClient, date) -> str | None:
    """One day's daily-note markdown, or None if there's no note that day.
    Prefers the Local REST API's dated periodic endpoint (format-agnostic); for
    today also tries the plain periodic route; falls back to the conventional
    '<Daily dir>/YYYY-MM-DD.md' path."""
    md = {"Accept": "text/markdown"}
    urls = [f"/periodic/daily/{date.year}/{date.month}/{date.day}/"]
    if date == datetime.now(_tz()).date():
        urls.insert(0, "/periodic/daily/")
    for url in urls:
        r = await c.get(url, headers=md)
        if r.status_code == 200:
            return r.text
    folder = settings.obsidian_daily_dir.strip("/")
    path = f"{folder}/{date.isoformat()}.md" if folder else f"{date.isoformat()}.md"
    r = await c.get(_vault(path), headers=md)
    return r.text if r.status_code == 200 else None


async def list_tasks(when: str = "today", include_done: bool = False) -> dict:
    """List to-do tasks (checkbox lines) from the daily notes for a scope.

    ``when`` is 'today' (default), 'yesterday', 'recent' (last 7 days), or a
    specific 'YYYY-MM-DD'. Returns the open (unchecked) tasks; set
    ``include_done`` to also return completed ones. Use this for any task/to-do
    question instead of a full-text search.
    """
    open_tasks: list[dict] = []
    done_tasks: list[dict] = []
    async with _client() as c:
        for date in _resolve_dates(when):
            text = await _read_daily(c, date)
            if not text:
                continue
            for line in text.split("\n"):
                m = _CHECKBOX.match(line)
                if not m:
                    continue
                item = m.group(3).strip()
                if not item:
                    continue
                entry = {"task": item, "date": date.isoformat()}
                (done_tasks if m.group(2).lower() == "x" else open_tasks).append(entry)
    result = {"when": when, "count": len(open_tasks), "open_tasks": open_tasks}
    if include_done:
        result["done_tasks"] = done_tasks
    return result


# Tool specs (Ollama / OpenAI function-calling format)

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "capture",
            "description": (
                "Add a line to today's daily note. By default it goes under the "
                "Notes section — use this for 'note this down' / 'add to today's "
                "note'. Set section='tasks' for a to-do (filed under Tasks as a "
                "checkbox) or section='goals' for a goal. Only use tasks or goals "
                "when the user explicitly says task/to-do or goal; otherwise notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The note, task, or goal text"},
                    "section": {
                        "type": "string",
                        "enum": ["notes", "tasks", "goals"],
                        "description": "Daily-note section; defaults to 'notes'. 'tasks' becomes a checkbox.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_list",
            "description": (
                "Add an item to a standing named checklist as a checkbox — use for 'add X "
                "to my shopping list' or 'put X on the grocery list'. Creates the list if it "
                "doesn't exist. NOT for a one-off to-do for today: that's capture with "
                "section='tasks'. Only use a list when the user names one ('… list')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The list's name, e.g. 'shopping'"},
                    "item": {"type": "string", "description": "The item to add"},
                },
                "required": ["name", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_list",
            "description": (
                "Start a new, empty named checklist, e.g. 'start a shopping list'. Only needed "
                "to make an empty list — add_to_list already creates the list on first use."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "The list's name"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_list",
            "description": "Read back a named checklist, e.g. \"what's on my shopping list?\".",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "The list's name"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": (
                "List to-do tasks from the daily notes. Use this for ANY question about "
                "tasks/to-dos — 'do I have any tasks', 'what's on my to-do list', 'any open "
                "tasks from yesterday'. Set when to 'today' (default), 'yesterday', 'recent' "
                "(last 7 days), or a specific 'YYYY-MM-DD'. Returns open (unchecked) tasks "
                "unless include_done is true. Prefer this over search_notes for tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "when": {
                        "type": "string",
                        "description": "Scope: 'today', 'yesterday', 'recent', or 'YYYY-MM-DD'. Defaults to today.",
                    },
                    "include_done": {
                        "type": "boolean",
                        "description": "Also include completed tasks (default false = only open).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_item",
            "description": (
                "Tick off a checklist item in a note — use whenever the user marks something "
                "done/watched/finished/bought or says 'mark it complete'. Pass the note's path "
                "(from a search_notes/read_note result, or a list's path) and the item text as "
                "it appears in the note. Set done=false to un-tick. You MUST call this to mark "
                "something complete — never say it's done without it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Vault path of the note with the checklist"},
                    "item": {"type": "string", "description": "The checklist item's text, as it appears in the note"},
                    "done": {"type": "boolean", "description": "true to check off (default), false to un-check"},
                },
                "required": ["path", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_item",
            "description": (
                "Delete an item from a checklist entirely — use for 'remove X', 'delete X', "
                "'take X off the list'. This is different from check_item (which only ticks it "
                "off). Pass the list's name or path and the item text. If more than one item "
                "matches, it returns candidates — ask the user which one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The list's name or vault path"},
                    "item": {"type": "string", "description": "The item to remove, as it appears in the list"},
                },
                "required": ["path", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_list",
            "description": (
                "Empty a checklist — use for 'clear my shopping list' (removes all items) or "
                "'clear the done items'/'remove the checked ones' (set checked_only=true). Keeps "
                "the list itself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The list's name"},
                    "checked_only": {"type": "boolean", "description": "true = remove only checked items; false = remove all (default)"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Search the user's notes by keyword and get matching files with "
                "context. Use this FIRST to answer any question about their notes, "
                "then read_note on the most relevant result before answering. Never "
                "invent note contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keywords to search for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "Read the full contents of one note by its vault path (e.g. 'Projects/Roof/Roof.md').",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Vault-relative path to the note"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_note",
            "description": "Create a new note (or overwrite one) at a vault path with the given markdown content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Vault-relative path, e.g. 'Ideas/App concept'"},
                    "content": {"type": "string", "description": "Markdown body"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_note",
            "description": "Append markdown to an existing note (created if it doesn't exist yet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Vault-relative path to the note"},
                    "content": {"type": "string", "description": "Markdown to append"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": (
                "Start a new project: creates a folder under the Projects area seeded with a "
                "README landing note. Use when the user says 'create a new project named X'. "
                "To put documents in it, call add_project_doc for each one — do NOT also "
                "create a note named after the project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name, e.g. 'Kitchen remodel'"},
                    "description": {"type": "string", "description": "Optional one-line description"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_project_doc",
            "description": "Add a document (note) to an existing project's folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Existing project name"},
                    "title": {"type": "string", "description": "Document title / filename"},
                    "content": {"type": "string", "description": "Markdown body"},
                },
                "required": ["project", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "List the user's existing projects (folders under the Projects area).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_DISPATCH = {
    "capture": capture,
    "create_list": create_list,
    "add_to_list": add_to_list,
    "read_list": read_list,
    "list_tasks": list_tasks,
    "check_item": check_item,
    "remove_item": remove_item,
    "clear_list": clear_list,
    "search_notes": search_notes,
    "read_note": read_note,
    "create_note": create_note,
    "append_note": append_note,
    "create_project": create_project,
    "add_project_doc": add_project_doc,
    "list_projects": list_projects,
}


async def execute_tool(name: str, arguments: dict) -> dict:
    """Run a notes tool by name. Never raises - errors come back as data."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return await fn(**(arguments or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except httpx.HTTPError as exc:
        return {"error": f"obsidian request failed (is your notes machine awake?): {exc}"}


async def health() -> bool:
    if not settings.obsidian_enabled:
        return False
    try:
        async with _client() as c:
            # Short probe timeout: the vault REST API runs on a separate host that
            # may be asleep/offline, in which case a connect black-holes for the
            # full client timeout (15s). /api/health fans this out and is polled
            # every 20s - an offline vault must not hang the health check.
            resp = await c.get("/", timeout=5)
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
