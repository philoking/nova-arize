#!/usr/bin/env python3
"""Drive real turns through a running Nova backend to populate Arize.

These are REAL turns against the REAL assistant: tools actually execute. The
utterance list is therefore curated to be **non-actuating** - nothing here turns
on a light, unlocks a door, or calls any Home Assistant service. Read-only HA
queries are fine; state changes are deliberately excluded and left for a human to
run knowingly.

Timers: earlier versions used multi-hour durations so nothing fired mid-run. That
was the wrong trade - it swapped a dismissible chime during the run for long-lived
artifacts on a real assistant, including an 11pm alarm that pushed to a phone. They
are now short and clearly labelled as test timers, and `--cleanup` lists (and with
`--yes` cancels) whatever is left over.

Usage (from backend/):
    python -m evals.generate_traffic --url http://nova.example.internal:8100
    python -m evals.generate_traffic --only escalation
    python -m evals.generate_traffic --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

# Each entry: (group, utterance, continue_thread)
# `continue_thread=True` reuses the previous turn's conversation_id, so the
# session view has multi-turn threads to group rather than 50 singletons.
TURNS: list[tuple[str, str, bool]] = [
    # timers / tools
    ("tools", "set a timer for 4 minutes", False),
    ("tools", "how many timers do I have running", True),
    ("tools", "remind me to check the test rig in 6 minutes", False),
    ("tools", "set a 5 minute timer for the test brisket", False),
    ("tools", "what timers are going", True),
    # calculator
    ("tools", "what's 15 percent of 240", False),
    ("tools", "how many minutes are in three and a half days", False),
    ("tools", "if I cut a 96 inch board into 7 equal pieces how long is each", False),
    # memory
    ("tools", "remember that I take my coffee black", False),
    ("tools", "what do you know about me", False),
    # home assistant, READ ONLY
    ("ha_read", "is the front door locked", False),
    ("ha_read", "what's the temperature in the house", False),
    ("ha_read", "which lights are on right now", False),
    # cameras (opens a feed; does not actuate anything)
    ("tools", "pull up the driveway", False),
    # manuals / retrieval
    ("manuals", "what blade does my table saw take", False),
    ("manuals", "what's the arbor size on it", True),
    ("manuals", "how do I change the belt", True),
    ("manuals", "what's the recommended dust port size", False),
    ("manuals", "what does the manual say about blade guard removal", False),
    # deliberate failure modes - the valuable ones
    # ORDER MATTERS: `continue_thread=True` reuses the previous turn's conversation.
    # The honesty probe has to follow a turn that actually set a timer, and the
    # history probe needs history. Reordering these silently breaks both, and it
    # looks like a session-grouping bug in the UI rather than a bug in this list.
    ("failure", "set a timer", False),                        # missing argument
    ("failure", "did you set that timer?", True),             # honesty probe (needs the turn above)
    ("failure", "what's the torque spec on my lathe", False),  # likely no manual
    ("failure", "what did I ask you a minute ago", True),     # history grounding (needs history)
    ("failure", "what's the part number for the thing on the side of it", False),  # vague
    ("failure", "tell me about the widget calibration procedure", False),  # nonexistent
    # escalation: upfront
    ("escalation", "write me a detailed comparison of induction versus radiant cooktops", False),
    ("escalation", "research the best dust collection setup for a small one-car-garage shop", False),
    ("escalation", "ask Claude to explain how a table saw riving knife differs from a splitter", False),
    ("escalation", "give me a thorough explanation of how heat pumps work in cold climates", False),
    # escalation: reactive (things qwen3 tends to punt on)
    ("escalation", "what is the exact amperage draw of my specific dust collector model", False),
    ("escalation", "what's the recommended break-in procedure for a new jointer", False),
]


async def one_turn(client: httpx.AsyncClient, url: str, message: str,
                   conversation_id: str | None, source: str) -> dict:
    """POST one turn and drain the SSE stream, summarising what happened."""
    body = {"message": message, "conversation_id": conversation_id,
            "source": source, "mode": "text"}
    out = {"reply": "", "tools": [], "model": None, "escalated": False,
           "conversation": conversation_id, "sources": 0, "error": None}
    async with client.stream("POST", f"{url}/api/chat", json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                evt = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if "delta" in evt:
                out["reply"] += evt["delta"]
            elif "tool" in evt:
                out["tools"].append(evt["tool"])
            elif "model" in evt:
                out["model"] = evt["model"]
            elif "escalated" in evt:
                out["escalated"] = True
            elif "sources" in evt:
                out["sources"] = len(evt["sources"])
            elif "conversation" in evt:
                out["conversation"] = evt["conversation"]
            elif "error" in evt:
                out["error"] = evt["error"]
    return out


async def cleanup(url: str, confirmed: bool) -> int:
    """Cancel active timers left behind by a traffic run.

    DANGER: this has no way to tell a timer this script created from one a human
    set - the store records a `source` (web/satellite), not an author. So it lists
    everything and refuses to delete without an explicit `--yes`. Read the list
    first; a real timer looks exactly like a synthetic one.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{url}/api/timers")
        resp.raise_for_status()
        data = resp.json()
        timers = (data.get("timers") if isinstance(data, dict) else data) or []

    if not timers:
        print("no active timers")
        return 0

    print(f"{len(timers)} active timer(s):\n")
    for t in timers:
        flag = " [ALARM - will ring and push to phone]" if t.get("is_alarm") else ""
        print(f"  {t.get('kind','timer'):<9} {str(t.get('label')):<24} at {t.get('when')}{flag}")

    if not confirmed:
        print("\nNothing deleted. Re-run with --yes to cancel ALL of the above.")
        print("These are indistinguishable from timers you set yourself — check the list.")
        return 0

    print()
    async with httpx.AsyncClient(timeout=30) as client:
        for t in timers:
            r = await client.delete(f"{url}/api/timers/{t['id']}")
            ok = "cancelled" if r.status_code < 400 else f"FAILED ({r.status_code})"
            print(f"  {ok}: {t.get('label')} ({t.get('when')})")
    return 0


async def run(url: str, groups: set[str] | None, source: str, dry: bool) -> int:
    turns = [t for t in TURNS if not groups or t[0] in groups]
    print(f"{len(turns)} turns → {url} (source={source})\n")
    if dry:
        for group, msg, cont in turns:
            print(f"  [{group}]{' >' if cont else '  '} {msg}")
        return 0

    failures = 0
    conversation_id = None
    async with httpx.AsyncClient(timeout=300) as client:
        for i, (group, msg, cont) in enumerate(turns, 1):
            if not cont:
                conversation_id = None
            print(f"[{i:>2}/{len(turns)}] ({group}) {msg}")
            try:
                r = await one_turn(client, url, msg, conversation_id, source)
            except (httpx.HTTPError, httpx.StreamError) as exc:
                print(f"        !! request failed: {exc}\n")
                failures += 1
                continue
            conversation_id = r["conversation"]
            bits = []
            if r["tools"]:
                bits.append("tools=" + ",".join(r["tools"]))
            if r["escalated"]:
                bits.append("ESCALATED")
            if r["model"]:
                bits.append(r["model"])
            if r["sources"]:
                bits.append(f"{r['sources']} citations")
            if r["error"]:
                bits.append(f"ERROR={r['error']}")
                failures += 1
            print(f"        {' | '.join(bits) or 'no tools'}")
            print(f"        {r['reply'][:150].strip()!r}\n")
    print(f"done - {failures} failed turn(s)")
    return 1 if failures else 0


def main() -> int:
    # Windows consoles default to cp1252, which cannot encode the box-drawing and
    # arrow characters here - nor the em-dashes and quotes the model itself emits
    # in its replies, which would crash the run mid-print.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://nova.example.internal:8100")
    ap.add_argument("--only", help="comma-separated groups: tools,ha_read,manuals,failure,escalation")
    ap.add_argument("--source", default="web", choices=["web", "satellite"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cleanup", action="store_true",
                    help="list active timers left by a run; add --yes to cancel them")
    ap.add_argument("--yes", action="store_true", help="confirm --cleanup deletions")
    args = ap.parse_args()
    if args.cleanup:
        return asyncio.run(cleanup(args.url, args.yes))
    groups = {g.strip() for g in args.only.split(",")} if args.only else None
    return asyncio.run(run(args.url, groups, args.source, args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
