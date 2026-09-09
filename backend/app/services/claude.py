"""Claude as the auto-escalation "big brain".

When the local model (qwen3) can't handle a turn - it struggles reactively, or the
request is one the router sends straight to Claude - the turn runs here instead.
Unlike the old manual, tool-free escalation lane (now retired, see #65), this drives
the **same routed local tools** qwen3 had via the Anthropic tool-use loop, so hard
multi-step turns actually complete. It's only ever reached when auto-escalation is
enabled (``auto_escalate`` + an API key); with it off, nothing leaves the LAN.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable

import anthropic

from .. import tracing
from ..config import settings
from ..settings_store import store

ToolExecutor = Callable[[str, dict], Awaitable[dict]]

# Server-tool style turns can pause after ~10 internal iterations (stop_reason
# "pause_turn"); we resume a bounded number of times.
_MAX_PAUSE_RESUMES = 4


def _client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=settings.claude_api_key)


def _payload(messages: list[dict]) -> list[dict]:
    """The message list sent to Claude - role + string content only. Drops anything
    that isn't a user/assistant text turn (the local history only holds those)."""
    out = []
    for turn in messages:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


# Adaptive thinking ({"type": "adaptive"}) is 4.6-and-later. Older models, notably
# claude-haiku-4-5 (our FAST tier), reject it with a 400 and take the earlier
# {"type": "enabled", "budget_tokens": N} form. Every reactive escalation was
# failing this way. The fast tier exists to be quick, so we omit thinking there.
_NO_ADAPTIVE_THINKING = ("claude-haiku-", "claude-3-", "claude-sonnet-4-5")


def _thinking_for(model: str) -> dict | None:
    """The ``thinking`` parameter this model accepts, or None to omit it."""
    name = (model or "").lower()
    if any(name.startswith(prefix) for prefix in _NO_ADAPTIVE_THINKING):
        return None
    return {"type": "adaptive"}


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert Ollama/OpenAI function specs to Anthropic tool specs. The namespaced
    name (``manuals__search_manuals``) satisfies Anthropic's ``[a-zA-Z0-9_-]`` name
    rule, so ``registry.execute`` can dispatch it unchanged."""
    out = []
    for spec in tools or []:
        fn = spec.get("function", {})
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


async def agent_reply(
    messages: list[dict],
    tools: list[dict] | None = None,
    execute_tool: ToolExecutor | None = None,
    system_note: str = "",
    response_mode: str = "voice",
    model: str | None = None,
    parent=None,
) -> AsyncIterator[dict]:
    """Claude as the escalation *brain*: run the agent loop with the local tools.

    Same event interface as ``llm.agent_reply`` (``delta`` / ``tool`` / ``error``)
    so the chat handler can consume either uniformly. Unlike ``stream_reply`` (the
    opt-in, tool-free, confirm-gated lane), this drives the same routed local tools
    qwen3 had - so it can search manuals, hit the web, control HA, etc. It is only
    reached when auto-escalation is enabled and a local turn struggled.

    ``model`` picks the tier: the fast/cheap model on the reactive path, the deeper
    model on the upfront path. Defaults to the deep ``claude_model``.
    """
    convo = _payload(messages)  # user/assistant text turns only
    if not convo:
        yield {"type": "error", "message": "nothing to escalate"}
        return

    prompt_key = "text_prompt" if response_mode == "text" else "voice_prompt"
    system = store.get(prompt_key) + (system_note or "")
    a_tools = _to_anthropic_tools(tools) if tools else None

    chosen_model = model or settings.claude_model
    kwargs: dict = {
        "model": chosen_model,
        "max_tokens": settings.claude_max_tokens,
        "system": system,
    }
    thinking = _thinking_for(chosen_model)
    if thinking:
        kwargs["thinking"] = thinking
    if a_tools:
        kwargs["tools"] = a_tools

    client = _client()
    # Bounded so a misbehaving loop can't run forever; generous for real chains.
    rounds = settings.max_tool_rounds + _MAX_PAUSE_RESUMES if a_tools else 1
    try:
        for _ in range(rounds + 1):
            # AnthropicInstrumentor parents its span off OTel's implicit current-span
            # context, while everything here uses explicit parents (see app/tracing.py).
            # Without activating the turn span, Claude's spans become separate roots.
            with tracing.activate(parent):
                async with client.messages.stream(messages=convo, **kwargs) as stream:
                    async for text in stream.text_stream:
                        yield {"type": "delta", "text": text}
                    final = await stream.get_final_message()

            if final.stop_reason == "tool_use" and execute_tool is not None:
                # Echo Claude's turn (thinking + tool_use blocks) back, then run
                # each tool and return the results as a user turn - the loop.
                convo.append({"role": "assistant", "content": final.content})
                results = []
                for block in final.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    yield {"type": "tool", "name": block.name, "arguments": dict(block.input or {})}
                    result = await execute_tool(block.name, dict(block.input or {}))
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
                convo.append({"role": "user", "content": results})
                continue

            if final.stop_reason == "pause_turn":
                # Server-tool style pause (rare with local tools) - resume.
                convo.append({"role": "assistant", "content": final.content})
                continue

            return  # end_turn / max_tokens - answered in words, already streamed.
    except anthropic.APIError as exc:
        yield {"type": "error", "message": f"claude error: {exc}"}
    finally:
        await client.close()
