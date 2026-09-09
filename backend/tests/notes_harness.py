#!/usr/bin/env python3
"""One-off notes/tasks tool harness (#27).

Exercises the Obsidian `notes` tools directly (no LLM) against the real vault to
validate what actually lands, phrasing by phrasing.

SAFETY (see #27):
- Everything it creates is prefixed ``nova-test-``.
- Every destructive/modify call (`check_item`/`remove_item`/`clear_list`) is given
  the EXACT path returned by a create call - never a fuzzy name - so the tools'
  vault-wide fuzzy fallback can never resolve onto a real note.
- Each destructive result's path is asserted to be inside the nova-test namespace;
  a violation hard-aborts before any further writes.
- Reads (`search_notes`/`read_note`/`list_tasks`/`list_projects`) run vault-wide -
  harmless.
- `capture` appends to today's daily note by design; its lines are tagged
  ``nova-test-`` (that note is disposable per #27). We don't rewrite the daily note.
- Cleans up its own note/list/project files at the end (exact nova-test paths).

NOT a unittest (name isn't ``test_*``), so ``unittest discover`` ignores it - it
must never run automatically. Run it deliberately:
    python -m tests.notes_harness      # from backend/, inside the container
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services import obsidian as O  # noqa: E402

PREFIX = "nova-test-"
_PASS: list[str] = []
_FAIL: list[str] = []
_created: list[str] = []  # exact vault paths we made, for cleanup
_daily_lines = 0          # nova-test lines added to today's daily note (reported, not cleaned)


def rec(name: str, ok: bool, detail: str = "") -> bool:
    (_PASS if ok else _FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def in_ns(path) -> bool:
    return isinstance(path, str) and PREFIX in path.lower()


def guard(res: dict, op: str) -> dict:
    """Hard-abort if a destructive op resolved OUTSIDE the nova-test namespace.
    Given we only ever pass exact nova-test paths, this should never fire - it's a
    tripwire, not normal control flow."""
    p = (res or {}).get("path", "")
    if res.get("ok") and p and not in_ns(p):
        raise SystemExit(f"ABORT — {op} resolved outside nova-test namespace: {p!r}")
    return res


async def _delete(path: str) -> None:
    async with O._client() as c:
        try:
            await c.delete(O._vault(path))
        except Exception as e:  # noqa: BLE001
            print(f"  (cleanup: couldn't delete {path}: {e})", flush=True)


def _track(res: dict) -> dict:
    p = res.get("path") or res.get("readme")
    if in_ns(p) and p not in _created:
        _created.append(p)
    return res


async def _read(path: str) -> str:
    r = await O.read_note(path)
    return r.get("content", "") if "content" in r else ""


# Lists: add / read / phrasing normalization / check / remove / clear
async def test_lists() -> None:
    print("\n── Lists ──", flush=True)
    # 1. create-on-first-add
    r = _track(await O.add_to_list("nova-test-groceries", "milk"))
    path = r.get("path", "")
    rec("add_to_list creates the list + adds 'milk'", r.get("ok") and in_ns(path), path or r.get("error", ""))
    if not path:
        return

    # 2. phrasing normalization - 'X', 'X list', 'X checklist' should be one file
    _track(await O.add_to_list("nova-test-groceries list", "eggs"))
    _track(await O.add_to_list("nova-test-groceries checklist", "bread"))
    body = await _read(path)
    same = all(x in body for x in ("Milk", "Eggs", "Bread"))
    rec("phrasing 'list'/'checklist' normalize to the same file", same,
        "milk+eggs+bread all present" if same else f"missing some in {path}")

    # 3. read_list by name resolves to that file
    rl = await O.read_list("nova-test-groceries")
    rec("read_list returns the list content", rl.get("path") == path and "Milk" in rl.get("content", ""),
        rl.get("path", rl.get("error", "")))

    # 4. check_item (tick) - EXACT path
    ci = guard(await O.check_item(path, "eggs", True), "check_item")
    body = await _read(path)
    rec("check_item ticks 'eggs'", ci.get("ok") and "- [x] Eggs" in body, ci.get("item", ci.get("error", "")))

    # 5. remove_item (delete) - EXACT path
    ri = guard(await O.remove_item(path, "milk"), "remove_item")
    body = await _read(path)
    rec("remove_item deletes 'milk' (keeps eggs/bread)",
        ri.get("ok") and "Milk" not in body and "Eggs" in body and "Bread" in body,
        ri.get("removed", ri.get("error", "")))

    # 6. ambiguous remove must NOT delete, must return candidates
    _track(await O.add_to_list("nova-test-groceries", "green apples"))
    _track(await O.add_to_list("nova-test-groceries", "red apples"))
    amb = guard(await O.remove_item(path, "apples"), "remove_item")
    body = await _read(path)
    still_both = "Green apples" in body and "Red apples" in body
    rec("ambiguous remove asks instead of guessing",
        amb.get("error") == "ambiguous" and still_both, str(amb.get("candidates", amb.get("error", ""))))

    # 7. clear checked-only, then clear all
    guard(await O.check_item(path, "bread", True), "check_item")
    cc = guard(await O.clear_list(path, checked_only=True), "clear_list")
    body = await _read(path)
    rec("clear_list(checked_only) removes only ticked items",
        cc.get("ok") and "Bread" not in body and "Green apples" in body, f"removed={cc.get('removed')}")
    ca = guard(await O.clear_list(path), "clear_list")
    body = await _read(path)
    checkboxes = [ln for ln in body.split("\n") if ln.strip().startswith(("- [", "* ["))]
    rec("clear_list empties the list (keeps the note)", ca.get("ok") and not checkboxes,
        f"removed={ca.get('removed')}, remaining checkboxes={len(checkboxes)}")


# Notes: create / append / read / search
async def test_notes() -> None:
    print("\n── Notes ──", flush=True)
    tok = "novatoken-alpha-42"
    cn = _track(await O.create_note("nova-test-note-alpha", f"# Nova Test Alpha\n\nfirst line {tok}\n"))
    rec("create_note writes a note", cn.get("ok") and in_ns(cn.get("path", "")), cn.get("path", cn.get("error", "")))

    an = _track(await O.append_note("nova-test-note-alpha", f"second line {tok}-b"))
    body = await _read("nova-test-note-alpha")
    rec("append_note adds to the note", an.get("ok") and tok in body and f"{tok}-b" in body,
        "both lines present" if tok in body else "append missing")

    sr = await O.search_notes(tok)
    hit = any("nova-test-note-alpha" in (h.get("path") or "") for h in sr.get("results", []))
    rec("search_notes finds the note by content", hit, f"{sr.get('count')} hits")

    rn = await O.read_note("nova-test-note-alpha")
    rec("read_note returns exact content", "content" in rn and tok in rn["content"], rn.get("path", rn.get("error", "")))


# Capture + tasks (today's daily note)
async def test_capture_and_tasks() -> None:
    global _daily_lines
    print("\n── Capture & tasks (today's daily note) ──", flush=True)
    note_txt = "nova-test capture note novatoken-n1"
    task_txt = "nova-test capture task novatoken-t1"
    goal_txt = "nova-test capture goal novatoken-g1"

    cn = await O.capture(note_txt, "notes")
    ct = await O.capture(task_txt, "tasks")
    cg = await O.capture(goal_txt, "goals")
    _daily_lines += sum(1 for r in (cn, ct, cg) if r.get("ok"))

    today = datetime.now(O._tz()).date()
    async with O._client() as c:
        daily = await O._read_daily(c, today) or ""
    rec("capture(notes) lands in today's daily note", cn.get("ok") and note_txt in daily, cn.get("section", cn.get("error", "")))
    rec("capture(goals) lands under Goals", cg.get("ok") and goal_txt in daily, cg.get("section", cg.get("error", "")))

    lt = await O.list_tasks("today")
    has_task = any(task_txt in t.get("task", "") for t in lt.get("open_tasks", []))
    rec("capture(tasks) shows up in list_tasks('today')", ct.get("ok") and has_task, f"{lt.get('count')} open tasks")

    ltd = await O.list_tasks("today", include_done=True)
    rec("list_tasks(include_done) returns done_tasks", "done_tasks" in ltd, "has done_tasks key")


# Projects
async def test_projects() -> None:
    print("\n── Projects ──", flush=True)
    cp = _track(await O.create_project("nova-test-project", "a throwaway test project"))
    readme = cp.get("readme", "")
    rec("create_project makes the folder + README", cp.get("ok") and in_ns(readme), readme or cp.get("error", ""))

    pd = _track(await O.add_project_doc("nova-test-project", "nova-test-doc", "doc body novatoken-d1"))
    rec("add_project_doc adds a doc", pd.get("ok") and in_ns(pd.get("path", "")), pd.get("path", pd.get("error", "")))

    lp = await O.list_projects()
    hit = any("nova-test-project" in p for p in lp.get("projects", []))
    rec("list_projects lists the project", hit, f"{lp.get('count')} projects")


async def cleanup() -> None:
    print("\n── Cleanup ──", flush=True)
    for p in _created:
        await _delete(p)
        print(f"  deleted {p}", flush=True)
    if _daily_lines:
        print(f"  NOTE: {_daily_lines} nova-test line(s) were added to today's daily note "
              f"(left in place — it's disposable; delete/recreate to clear).", flush=True)


async def main() -> int:
    print(f"Notes harness (#27) — vault {O.settings.obsidian_url}", flush=True)
    if not await O.health():
        print("ABORT — Obsidian REST API not reachable (is the Windows machine awake + plugin running?)", flush=True)
        return 2
    try:
        await test_lists()
        await test_notes()
        await test_capture_and_tasks()
        await test_projects()
    finally:
        await cleanup()

    print(f"\n{'='*48}\nRESULT: {len(_PASS)} passed, {len(_FAIL)} failed", flush=True)
    if _FAIL:
        print("FAILED: " + ", ".join(_FAIL), flush=True)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
