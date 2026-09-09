#!/usr/bin/env python3
"""Smoke test: prove a span reaches Arize AX before wiring the live app.

This emits the exact shape ``/api/chat`` produces - an AGENT root for the turn
with a CHAIN child for the router - using the same ``app.tracing`` helpers the
real path uses. That matters because the interesting risk isn't "does the
exporter work", it's "does an explicitly-parented span tree survive when the
parent is held open across an async generator's yields". This reproduces that
without needing Nova's Ollama/HA/Qdrant services to be reachable.

Usage (from backend/), with the two values from the Arize onboarding screen:

    # PowerShell
    $env:ARIZE_SPACE_ID="…"; $env:ARIZE_API_KEY="…"; python -m evals.arize_smoke

    # bash
    ARIZE_SPACE_ID=… ARIZE_API_KEY=… python -m evals.arize_smoke

Pass ``--live`` to drive the real routed agent loop against Nova's Ollama (tool
calls are still faked, so nothing is timed or toggled) - that is what exercises
the LLM spans, real token counts and TTFT.

Then open the project in Arize. Exit code is non-zero if tracing never came up.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import tracing  # noqa: E402
from app.config import settings  # noqa: E402

# A stand-in turn. Deliberately one that routes to two skills in the real
# router, so the CHAIN span has something to say.
UTTERANCE = "set a timer for 10 minutes"
ROUTED_SKILLS = ["timers"]
TOOLS_OFFERED = [
    "timers__start_timer",
    "timers__set_reminder",
    "timers__list_timers",
    "timers__cancel_timer",
]
REPLY = "Timer set for 10 minutes."


async def one_turn() -> None:
    """Mirror api_chat: root span opened outside the stream, closed after it."""
    turn = tracing.start(
        "chat.turn",
        kind=tracing.AGENT,
        **{tracing.INPUT_VALUE: UTTERANCE, tracing.SESSION_ID: "smoke-conversation"},
    )

    with tracing.span("route", kind=tracing.CHAIN, parent=turn.raw) as route:
        route.set_input([UTTERANCE])
        route.set_output(ROUTED_SKILLS)
        route.set(**{tracing.METADATA: {
            "skills": ROUTED_SKILLS,
            "tools_offered": TOOLS_OFFERED,
        }})

    # Stand in for the streaming generator: the root span stays open across
    # awaits/yields while chunks go out, exactly as it does in the real handler.
    async def stream():
        for chunk in REPLY.split():
            await asyncio.sleep(0.02)
            yield chunk

    try:
        async for _chunk in stream():
            pass
    finally:
        turn.set_output(REPLY)
        turn.set(**{tracing.METADATA: {
            "source": "smoke",
            "mode": "voice",
            "model": settings.model,
            "escalated": False,
            "tool_calls": ["start_timer"],
        }})
        turn.end()


async def live_turn() -> None:
    """Drive the REAL routed agent loop against Nova's Ollama.

    Same shape as a browser turn - router, then ``llm.agent_reply`` with the
    routed tools - but the tool executor is faked, so nothing is actually timed
    or toggled. This is what proves the LLM spans: real token counts from
    Ollama's final chunk, a real TTFT, and real tool-call arguments.
    """
    from app.plugins.registry import Registry
    from app.plugins.timers import TimersProvider
    from app.services import llm

    registry = Registry([TimersProvider()])
    skills = registry.route([UTTERANCE])
    tools = registry.tools_for(skills)
    note = registry.context_for(skills)
    print(f"routed to {skills or 'nothing'} ({len(tools or [])} tools)")

    turn = tracing.start(
        "chat.turn",
        kind=tracing.AGENT,
        **{tracing.INPUT_VALUE: UTTERANCE, tracing.SESSION_ID: "smoke-live"},
    )
    with tracing.span("route", kind=tracing.CHAIN, parent=turn.raw) as route:
        route.set_input([UTTERANCE])
        route.set_output(skills or [])

    calls: list[str] = []

    async def fake_execute(name, args):
        # Mirrors the TOOL span main.py::api_chat creates around registry.execute.
        # Duplicated so this can check span shape without standing up the app.
        calls.append(name)
        print(f"  tool: {name}({args})")
        bare = name.split("__", 1)[-1]
        with tracing.span(f"tool.{bare}", kind=tracing.TOOL, parent=turn.raw) as tspan:
            tspan.set(**{tracing.TOOL_NAME: name, tracing.TOOL_PARAMETERS: args})
            result = {"ok": True, "timer": {"label": "timer", "in_seconds": 600,
                                            "when": "12:10 PM"}}
            tspan.set_output(result)
        return result

    reply: list[str] = []
    try:
        async for event in llm.agent_reply(
            [{"role": "user", "content": UTTERANCE}],
            tools=tools, execute_tool=fake_execute, system_note=note,
            response_mode="voice", parent=turn.raw,
        ):
            if event["type"] == "delta":
                reply.append(event["text"])
            elif event["type"] == "error":
                print(f"  ERROR: {event['message']}", file=sys.stderr)
    finally:
        text = "".join(reply).strip()
        turn.set_output(text)
        turn.set(**{tracing.METADATA: {
            "source": "smoke-live", "mode": "voice", "model": settings.model,
            "tool_calls": calls,
        }})
        turn.end()
    print(f"reply: {text!r}")


def main() -> int:
    live = "--live" in sys.argv
    if not settings.arize_enabled:
        print("ARIZE_SPACE_ID / ARIZE_API_KEY are not set — nothing to do.", file=sys.stderr)
        print("See the header of this file for the exact command.", file=sys.stderr)
        return 2

    started = time.monotonic()
    if not tracing.init():
        print("tracing.init() failed — see the log line above.", file=sys.stderr)
        return 1
    print(f"registered in {time.monotonic() - started:.2f}s "
          f"(project={settings.arize_project})")

    if live:
        print(f"live mode — driving {settings.llm_url} with model {settings.model}")
        asyncio.run(live_turn())
        print("emitted 1 trace: chat.turn [AGENT] → route [CHAIN] + llm.local [LLM] ×N")
    else:
        asyncio.run(one_turn())
        print("emitted 1 trace: chat.turn [AGENT] → route [CHAIN]")

    # The exporter batches. Flush explicitly, or a short-lived script exits
    # before anything is actually sent - a genuinely easy way to conclude
    # "tracing is broken" when it is merely asynchronous.
    flushed = False
    try:
        provider = tracing._provider
        if provider is not None and hasattr(provider, "force_flush"):
            flushed = bool(provider.force_flush(timeout_millis=10_000))
    except Exception as exc:  # noqa: BLE001
        print(f"force_flush raised: {exc}", file=sys.stderr)
    print(f"force_flush: {'ok' if flushed else 'not confirmed'}")
    print("\nOpen Arize → your project. The trace should appear within a few seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
