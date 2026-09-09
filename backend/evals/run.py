#!/usr/bin/env python3
"""Agent eval runner (#60) - drive cases.jsonl through the REAL routed agent loop
and dump prompt / response / tool-calls to CSV for inspection.

This is the seed of the Tier-2 harness described in #60. It does NOT score
grounding yet - it records what actually happened per case (routed skills, the
exact system prompt the model saw, its final reply, and every tool it called with
args), plus a quick tool-selection match, so you can eyeball the run in a sheet.

Nothing real is touched: tool calls go to a fake executor that returns each case's
`fixture` (the pure calculator is run for real so math answers are genuine). HA /
Obsidian / web providers are force-enabled for the run regardless of local config,
because the point is to test the model's tool *selection*, not this box's setup.

Run (from backend/, needs Nova's Ollama reachable):
    python -m evals.run                      # all cases -> evals/run_output.csv
    python -m evals.run --limit 5            # first 5
    python -m evals.run --model qwen3:14b    # compare a model
    python -m evals.run --out /tmp/x.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from app import memory, skills, timers  # noqa: E402
from app.config import settings  # noqa: E402
from app.plugins.calculator import CalculatorProvider  # noqa: E402
from app.plugins.home_assistant import HomeAssistantProvider  # noqa: E402
from app.plugins.memory import MemoryProvider  # noqa: E402
from app.plugins.obsidian import ObsidianProvider  # noqa: E402
from app.plugins.registry import Registry  # noqa: E402
from app.plugins.timers import TimersProvider  # noqa: E402
from app.plugins.websearch import WebSearchProvider  # noqa: E402
from app.services import home_assistant as ha  # noqa: E402
from app.services import llm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = settings and None  # set in main()

# time freezing
_REAL_DT = _dt.datetime


class _Frozen(_dt.datetime):
    """datetime whose .now() returns a pinned instant (per-case), else real now."""
    pinned = None

    @classmethod
    def now(cls, tz=None):
        if cls.pinned is None:
            return _REAL_DT.now(tz)
        return cls.pinned if tz is None else cls.pinned.astimezone(tz)


# Patch the `datetime` the time-sensitive modules imported by name.
skills.datetime = _Frozen
import app.plugins.timers as _pt  # noqa: E402
import app.plugins.calculator as _pc  # noqa: E402
_pt.datetime = _Frozen
_pc.datetime = _Frozen
_REAL_NOW = timers._now
timers._now = lambda: int((_Frozen.pinned or _REAL_DT.now(timers._tz())).timestamp())


def _pin_time(now_iso):
    if not now_iso:
        _Frozen.pinned = None
        return
    dt = _dt.datetime.fromisoformat(now_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timers._tz())
    _Frozen.pinned = dt


# force-enable providers so every skill is offered, regardless of local env
class _Forced:
    def __init__(self, prov, enabled):
        self._p, self.name, self._enabled = prov, prov.name, enabled

    @property
    def enabled(self):
        return self._enabled

    def tool_specs(self):
        return self._p.tool_specs()

    def system_note(self):
        return self._p.system_note()

    async def execute(self, tool, args):
        return await self._p.execute(tool, args)

    async def health(self):
        return True


_CALC = CalculatorProvider()
_BASE = {
    "timers": TimersProvider(), "calculator": _CALC, "memory": MemoryProvider(),
    "notes": ObsidianProvider(), "home_assistant": HomeAssistantProvider(),
    "web": WebSearchProvider(),
}


def _registry_for(state):
    ha_on = bool(state.get("ha_devices")) and state.get("ha_enabled", True) is not False
    forced = [_Forced(p, p.name != "home_assistant" or ha_on) for p in _BASE.values()]
    return Registry(forced)


def _apply_state(case):
    st = case.get("state", {})
    _pin_time(case.get("now"))
    ha.devices = lambda: list(st.get("ha_devices") or [])
    timers.ringing = lambda: list(st.get("ringing") or [])
    mems = [{"id": i, "text": t, "tier": "long", "source": "user", "created_at": 0, "expires_at": None}
            for i, t in enumerate(st.get("memories") or [])]
    memory.list_active = lambda limit=None: list(mems)
    memory.prune = lambda: None


# temperature-0 stream (patched over llm._stream_once for reproducibility)
async def _stream0(messages, tools):
    payload = {"model": MODEL, "messages": messages, "think": settings.think,
               "stream": True, "options": {"temperature": 0}}
    if tools:
        payload["tools"] = tools
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        async with client.stream("POST", f"{settings.llm_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


async def run_case(case):
    _apply_state(case)
    reg = _registry_for(case.get("state", {}))
    prior = case.get("prior_turns", [])
    user_texts = [t["content"] for t in prior if t.get("role") == "user"] + [case["utterance"]]
    routed = reg.route(user_texts)
    tools = reg.tools_for(routed)
    note = reg.context_for(routed)
    history = [t for t in prior if t.get("role") in ("user", "assistant")] + \
              [{"role": "user", "content": case["utterance"]}]

    system_prompt = llm.build_messages(history, system_note=note, response_mode="voice",
                                        has_tools=bool(tools))[0]["content"]

    calls = []

    async def fake_exec(name, args):
        calls.append({"name": name, "args": args})
        if name.startswith("calculator__"):          # pure + safe → run for real
            return await _CALC.execute(name.split("__", 1)[1], args)
        return case.get("fixture", {"ok": True})

    raw = ""
    async for ev in llm.agent_reply(history, tools, fake_exec, system_note=note, response_mode="voice"):
        if ev["type"] == "delta":
            raw += ev["text"]
        elif ev["type"] == "error":
            raw += f"\n[ERROR] {ev['message']}"
    return routed, tools, system_prompt, calls, raw


def _tool_verdict(case, called):
    exp = case["expect"]
    names = [c["name"] for c in called]
    if exp.get("no_tool"):
        return "PASS" if not called else f"FAIL (called {names})"
    want = exp.get("tool")
    if want:
        return "PASS" if want in names else f"FAIL (got {names or 'none'})"
    anyw = exp.get("tools_any")
    if anyw:
        return "PASS" if any(w in names for w in anyw) else f"FAIL (got {names or 'none'})"
    return ""


COLS = ["id", "tags", "utterance", "routed_skills", "expected_skill", "skill_ok",
        "tools_called", "expected_tool", "tool_ok", "num_tool_calls", "response",
        "think_leak", "tools_offered", "error", "system_prompt"]


async def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "run_output.csv"))
    args = ap.parse_args()
    MODEL = args.model
    llm._stream_once = _stream0  # temp-0 reproducible run

    cases = []
    for line in open(os.path.join(HERE, "cases.jsonl")):
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    for i, case in enumerate(cases, 1):
        cid = case["id"]
        print(f"[{i}/{len(cases)}] {cid}: {case['utterance'][:52]}", file=sys.stderr, flush=True)
        err = ""
        routed, tools, sysp, calls, raw = [], None, "", [], ""
        try:
            routed, tools, sysp, calls, raw = await run_case(case)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            print(f"    ! {err}", file=sys.stderr, flush=True)

        exp = case["expect"]
        want_skill = exp.get("skill", "")
        skill_ok = ("PASS" if (not routed if want_skill is None else want_skill in routed)
                    else f"FAIL (routed {routed})")
        rows.append({
            "id": cid,
            "tags": ",".join(case.get("tags", [])),
            "utterance": case["utterance"],
            "routed_skills": ",".join(routed),
            "expected_skill": "∅" if want_skill is None else (want_skill or ""),
            "skill_ok": skill_ok,
            "tools_called": json.dumps(calls, ensure_ascii=False),
            "expected_tool": exp.get("tool") or ",".join(exp.get("tools_any", []))
                             or ("<no_tool>" if exp.get("no_tool") else ""),
            "tool_ok": _tool_verdict(case, calls),
            "num_tool_calls": len(calls),
            "response": llm.strip_think(raw),
            "think_leak": "YES" if "<think>" in raw.lower() else "",
            "tools_offered": ",".join(t["function"]["name"] for t in (tools or [])),
            "error": err,
            "system_prompt": sysp,
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    sk = sum(1 for r in rows if r["skill_ok"] == "PASS")
    tl = sum(1 for r in rows if r["tool_ok"] == "PASS")
    tt = sum(1 for r in rows if r["tool_ok"] in ("PASS",) or r["tool_ok"].startswith("FAIL"))
    leaks = sum(1 for r in rows if r["think_leak"])
    print(f"\nWrote {len(rows)} rows -> {args.out}", file=sys.stderr)
    print(f"routing:  {sk}/{len(rows)} matched", file=sys.stderr)
    print(f"tool sel: {tl}/{tt} matched (of cases asserting a tool)", file=sys.stderr)
    print(f"<think> leaks: {leaks}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
