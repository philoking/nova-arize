"""LLM client - talks to Ollama's chat API and streams tokens back.

We use the native ``/api/chat`` endpoint (not the OpenAI shim) so we can pass
Ollama-specific options like ``think`` to switch off qwen3's reasoning trace,
and so tool-calling integrates cleanly with streaming.

``agent_reply`` runs the full agent loop: it streams assistant text, and when
the model emits tool calls it runs them (via the injected ``execute_tool``
callback), feeds the results back, and continues - until the model answers in
words. Callers receive a stream of typed events.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx

from .. import tracing
from ..config import settings
from ..settings_store import store

# Defensive: even with think disabled, strip any stray <think>…</think> block
# so reasoning never leaks into the spoken reply.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Appended to the system prompt whenever tools are available. Small models
# (qwen3:8b especially) tend to narrate a tool call ("let me search for that")
# and then stop without emitting one. This tells the model to act, not announce.
_TOOL_USE_NOTE = (
    " You have tools available. When a request needs one, call it immediately — "
    "do not announce that you are about to. Never reply with only 'let me check' "
    "or 'I'll search for that' and then stop; make the tool call instead. Say "
    "nothing until you have the tool's result, then answer from it."
)

# Signature of a "I'm about to do X" narration: a first-person intent phrase
# closely followed by an action verb, in a short turn. Used only as a fallback
# to detect when the model announced a tool it never actually invoked.
_TOOL_INTENT_RE = re.compile(
    r"\b(?:let me|let's|i'?ll|i will|i'?m going to|i am going to|i'?d|i can|"
    r"i'?m about to|going to|gonna)\b[^.?!]{0,40}\b(?:search|check|look|find|"
    r"pull|see|query|fetch|retrieve|use|access|scan|review|add|set|create|"
    r"start|mark|update|cancel|list|get|note|record|save)\b",
    re.IGNORECASE,
)

ToolExecutor = Callable[[str, dict], Awaitable[dict]]


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _announced_but_didnt_call(content: str) -> bool:
    """True if a tools-enabled turn produced no call but reads like a preamble.

    An empty reply, or a short one matching the "let me search…" signature, means
    the model meant to use a tool but narrated instead - worth one nudge. A long
    reply is treated as a genuine answer and left alone.
    """
    text = strip_think(content)
    if not text:
        return True
    if len(text) > 200:
        return False
    return bool(_TOOL_INTENT_RE.search(text))


def build_messages(
    history: list[dict],
    system_note: str = "",
    response_mode: str = "voice",
    has_tools: bool = False,
) -> list[dict]:
    """Prepend the system prompt to the caller-supplied turn history.

    ``response_mode`` selects the base prompt (terse-and-spoken vs. richer text);
    ``system_note`` is an extra fragment contributed by the tool providers;
    ``has_tools`` appends the call-don't-narrate directive when tools are live.
    """
    # The persona prompt is user-editable in Settings; the store returns the saved
    # override or the built-in default for this mode.
    prompt_key = "text_prompt" if response_mode == "text" else "voice_prompt"
    system = store.get(prompt_key) + system_note
    if has_tools:
        system += _TOOL_USE_NOTE
    messages = [{"role": "system", "content": system}]
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


async def _stream_once(messages: list[dict], tools: list[dict] | None) -> AsyncIterator[dict]:
    """Stream one model turn, yielding raw Ollama chunk dicts."""
    payload: dict = {
        "model": store.get("model"),
        "messages": messages,
        "think": settings.think,
        "stream": True,
        "options": {"temperature": 0.7},
    }
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


def _llm_span(parent, messages: list[dict], tools: list[dict] | None, round_label) -> object:
    """Open an LLM span for one model round.

    Started explicitly rather than via ``with``: this function's caller is an
    async generator that yields mid-round, and OTel's implicit context does not
    survive that (see ``tracing`` module docstring).
    """
    return tracing.start(
        "llm.local",
        kind=tracing.LLM,
        parent=parent,
        **{
            tracing.LLM_MODEL: store.get("model"),
            tracing.LLM_PROVIDER: "ollama",
            tracing.LLM_SYSTEM: "ollama",
            tracing.LLM_INVOCATION_PARAMS: {
                "temperature": 0.7,
                "think": settings.think,
                "stream": True,
                "tools": len(tools or []),
            },
            tracing.INPUT_VALUE: messages[-1].get("content", "") if messages else "",
            tracing.METADATA: {"round": round_label},
            **tracing.message_attributes(tracing.LLM_INPUT_MESSAGES, messages),
            **tracing.tool_attributes(tools),
        },
    )


def _close_llm_span(span, content: str, tool_calls: list[dict], ttft: float | None, done: dict) -> None:
    """Record the round's output, token counts and time-to-first-token.

    Ollama reports usage on the final streamed chunk as ``prompt_eval_count`` /
    ``eval_count``; there is no instrumentor to map those onto OpenInference's
    token attributes, so it is done by hand here.
    """
    prompt_tokens = done.get("prompt_eval_count")
    completion_tokens = done.get("eval_count")
    total = None
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total = prompt_tokens + completion_tokens
    out_msg = {"role": "assistant", "content": strip_think(content), "tool_calls": tool_calls}
    span.set(**{
        tracing.OUTPUT_VALUE: strip_think(content),
        tracing.LLM_TOKEN_PROMPT: prompt_tokens,
        tracing.LLM_TOKEN_COMPLETION: completion_tokens,
        tracing.LLM_TOKEN_TOTAL: total,
        tracing.LLM_TTFT_MS: round(ttft * 1000, 1) if ttft is not None else None,
        **tracing.message_attributes(tracing.LLM_OUTPUT_MESSAGES, [out_msg]),
    })
    span.end()


async def agent_reply(
    history: list[dict],
    tools: list[dict] | None = None,
    execute_tool: ToolExecutor | None = None,
    system_note: str = "",
    response_mode: str = "voice",
    parent=None,
) -> AsyncIterator[dict]:
    """Drive the conversation to a spoken answer, running tools as needed.

    ``parent`` is an optional tracing span handle; each model round is recorded
    as an LLM child of it. Passed explicitly because the turn's root span has to
    stay open across this generator's yields.

    Yields event dicts:
      {"type": "delta", "text": str} - a chunk of the assistant reply
      {"type": "tool",  "name": str, "arguments": dict} - a tool is being run
      {"type": "error", "message": str}
    """
    messages = build_messages(
        history, system_note=system_note, response_mode=response_mode, has_tools=bool(tools)
    )
    nudged = False

    for round_index in range(settings.max_tool_rounds if tools else 1):
        content = ""
        tool_calls: list[dict] = []
        span = _llm_span(parent, messages, tools, round_index)
        started = time.perf_counter()
        ttft: float | None = None
        done_chunk: dict = {}
        try:
            async for chunk in _stream_once(messages, tools):
                msg = chunk.get("message", {})
                delta = msg.get("content", "")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    content += delta
                    yield {"type": "delta", "text": delta}
                if msg.get("tool_calls"):
                    if ttft is None:  # a tool call is a "first token" too
                        ttft = time.perf_counter() - started
                    tool_calls.extend(msg["tool_calls"])
                if chunk.get("done"):
                    done_chunk = chunk
                    break
        except httpx.HTTPError as exc:
            span.record_error(exc)
            span.end()
            yield {"type": "error", "message": f"llm upstream error: {exc}"}
            return
        _close_llm_span(span, content, tool_calls, ttft, done_chunk)

        if not tool_calls or execute_tool is None:
            # No structured call. If the model only announced one, nudge it once;
            # the correction is internal and never streamed. Otherwise it answered
            # in words and we're done.
            if tools and execute_tool is not None and not nudged and _announced_but_didnt_call(content):
                nudged = True
                messages.append({
                    "role": "user",
                    "content": (
                        "You said you would use a tool but did not call one. If the "
                        "request needs a tool, call it now without replying in words "
                        "first. If no tool is needed, give your final answer."
                    ),
                })
                continue
            return  # model answered in words; already streamed.

        # Record the assistant's tool-call turn, then run each call.
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            yield {"type": "tool", "name": name, "arguments": args}
            result = await execute_tool(name, args)
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})

    # Ran out of tool rounds - ask once more without tools to force a final answer.
    # Tagged `round="final"` so exhausted loops are findable in Arize: hitting this
    # at all means the model burned every round without answering.
    span = _llm_span(parent, messages, None, "final")
    started = time.perf_counter()
    ttft: float | None = None
    content = ""
    done_chunk: dict = {}
    try:
        async for chunk in _stream_once(messages, None):
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - started
                content += delta
                yield {"type": "delta", "text": delta}
            if chunk.get("done"):
                done_chunk = chunk
                break
    except httpx.HTTPError as exc:
        span.record_error(exc)
        span.end()
        yield {"type": "error", "message": f"llm upstream error: {exc}"}
        return
    _close_llm_span(span, content, [], ttft, done_chunk)


async def list_models() -> list[str]:
    """Installed Ollama model names, for the Settings model picker. Empty if the
    LLM is unreachable (the caller falls back to the current selection)."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.llm_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError):
        return []
    return sorted(m["name"] for m in data.get("models", []) if m.get("name"))


async def health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.llm_url}/api/tags")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
