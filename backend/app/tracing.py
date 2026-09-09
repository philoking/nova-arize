"""Arize AX tracing for the Nova agent loop (OpenTelemetry / OpenInference).

Why this module exists in this shape
------------------------------------
None of Arize's 30+ advertised integrations cover how Nova is built. There is no
agent framework to auto-instrument, and the local model is reached over raw
``httpx`` against Ollama's native ``/api/chat`` (a deliberate choice - it lets us
pass ``think: false``), so ``openinference-instrumentation-ollama``, which patches
the ``ollama`` *client library*, has nothing to patch. Exactly one thing from the
quickstart applies to us for free: ``services/claude.py`` uses the ``anthropic``
SDK, so ``AnthropicInstrumentor`` traces the escalation leg with no code change.
Everything else - the turn root, the router, the local model, tool dispatch, the
manuals retriever, the escalation decision - is hand-rolled here.

Design rules
------------
1. **Inert unless configured.** No ``ARIZE_SPACE_ID``/``ARIZE_API_KEY``, or the
   packages simply aren't installed, and every helper degrades to a no-op. The
   test suite is stdlib-only and hermetic; it must pass with none of the OTel
   packages present, so nothing is imported at module scope.

2. **Explicit parents, not implicit context.** OTel's usual
   ``start_as_current_span`` relies on a contextvar, which does not survive an
   async generator's ``yield`` points - and ``/api/chat`` streams its whole turn
   out of one. So spans are created with an explicit ``parent=`` and the caller
   holds the handle. Less magic, but it actually works across a streaming
   response.

3. **Never break a turn.** A tracing failure is logged and swallowed. Observing
   the assistant must not be able to take the assistant down.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

from .config import settings

log = logging.getLogger(__name__)

# OpenInference semantic conventions
# Literals rather than an `openinference-semantic-conventions` import, so this
# module has no import-time dependency and the hermetic tests run without OTel.
SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
# `input.value` / `output.value` are strings with a companion mime type, not
# structures. Passing a list gives an OTel array attribute that renders as blank.
INPUT_MIME = "input.mime_type"
OUTPUT_MIME = "output.mime_type"
MIME_TEXT = "text/plain"
MIME_JSON = "application/json"
SESSION_ID = "session.id"
METADATA = "metadata"
TAG_TAGS = "tag.tags"

LLM_MODEL = "llm.model_name"
LLM_PROVIDER = "llm.provider"
LLM_SYSTEM = "llm.system"
LLM_INVOCATION_PARAMS = "llm.invocation_parameters"
LLM_INPUT_MESSAGES = "llm.input_messages"
LLM_OUTPUT_MESSAGES = "llm.output_messages"
LLM_TOOLS = "llm.tools"
LLM_TOKEN_PROMPT = "llm.token_count.prompt"
LLM_TOKEN_COMPLETION = "llm.token_count.completion"
LLM_TOKEN_TOTAL = "llm.token_count.total"

# Non-standard: OpenInference has no TTFT attribute. For a voice assistant it is
# the latency that matters, since the answer is spoken as it arrives.
LLM_TTFT_MS = "llm.time_to_first_token_ms"

TOOL_NAME = "tool.name"
TOOL_PARAMETERS = "tool.parameters"

RETRIEVAL_DOCUMENTS = "retrieval.documents"

# Span kinds we use. RETRIEVER is for the manuals/Qdrant search.
AGENT = "AGENT"
CHAIN = "CHAIN"
LLM = "LLM"
TOOL = "TOOL"
RETRIEVER = "RETRIEVER"

_tracer = None
_provider = None
_live = False


def enabled() -> bool:
    """True once a real exporter is wired up. Everything degrades gracefully
    when this is False, so callers never need to branch on it."""
    return _live


def init() -> bool:
    """Register the Arize exporter and attach the one auto-instrumentor that
    applies to us. Idempotent; returns whether tracing actually came up."""
    global _tracer, _provider, _live
    if _live:
        return True
    if not settings.arize_enabled:
        log.info("arize tracing off (no ARIZE_SPACE_ID / ARIZE_API_KEY)")
        return False

    try:
        from arize.otel import register
    except ImportError:
        # Configured but not installed - a deploy that got the env vars but not
        # the requirements. Warn loudly and carry on unobserved.
        log.warning("arize tracing configured but `arize-otel` is not installed — staying inert")
        return False

    try:
        _provider = register(
            space_id=settings.arize_space_id,
            api_key=settings.arize_api_key,
            project_name=settings.arize_project,
        )
        _tracer = _provider.get_tracer(__name__)
        _live = True
        log.info("arize tracing on — project=%s", settings.arize_project)
    except Exception as exc:  # noqa: BLE001 - tracing must never break startup
        log.warning("arize tracing failed to register: %s", exc)
        return False

    # The one free win from the quickstart: services/claude.py drives the
    # `anthropic` SDK, so the escalation leg instruments itself. Note the
    # asymmetry: the minority path gets better spans than the model doing most turns.
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor()
        instrumentor.instrument(tracer_provider=_provider)
        # instrument() does not raise on a version mismatch: it logs one
        # `DependencyConflict` line and leaves the SDK untouched. A silent no-op
        # here makes the escalation leg invisible, so verify attachment.
        if getattr(instrumentor, "is_instrumented_by_opentelemetry", False):
            log.info("arize: anthropic auto-instrumentation attached")
        else:
            log.warning(
                "arize: anthropic instrumentor did NOT attach — escalation turns "
                "will be untraced. Usually a version conflict; the instrumentor "
                "requires %s",
                list(instrumentor.instrumentation_dependencies()),
            )
    except ImportError:
        log.info("arize: openinference-instrumentation-anthropic absent; claude leg untraced")
    except Exception as exc:  # noqa: BLE001
        log.warning("arize: anthropic instrumentation failed: %s", exc)

    return True


def shutdown(timeout_ms: int = 5000) -> None:
    """Drain buffered spans before the process exits.

    Belt-and-braces, NOT the primary mechanism - I initially claimed this was
    fixing silent data loss on every deploy, and that was wrong. OpenTelemetry's
    `TracerProvider` defaults to `shutdown_on_exit=True` and `arize.otel.register()`
    does not override it, so a clean interpreter exit already flushes via `atexit`.
    `docker compose up -d` sends SIGTERM, uvicorn exits gracefully, and that path
    works.

    What this adds is narrower: a deterministic flush at FastAPI shutdown, before
    the event loop tears down, and cover for the case where the container is
    SIGKILLed after the grace period - where `atexit` never runs at all.
    """
    if not _live or _provider is None:
        return
    try:
        if hasattr(_provider, "force_flush"):
            _provider.force_flush(timeout_millis=timeout_ms)
        if hasattr(_provider, "shutdown"):
            _provider.shutdown()
    except Exception as exc:  # noqa: BLE001 - shutdown must never raise
        log.warning("arize: flush on shutdown failed: %s", exc)


def _coerce(value: Any) -> Any:
    """OTel attributes may only be scalars or homogeneous sequences of scalars.
    Anything richer (our tool args, routed skill lists, retrieval hits) is
    serialised to JSON so it survives to the UI intact."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def message_attributes(prefix: str, messages: list[dict]) -> dict[str, Any]:
    """Flatten a chat history into OpenInference's indexed message attributes.

    The convention is positional and dotted (``llm.input_messages.0.message.role``)
    with no published helper, so this is hand-rolled from the spec. Ollama's
    ``tool_calls`` shape is mapped onto the keys OpenAI-flavoured instrumentors
    emit, so both legs of the agent read alike in the UI.
    """
    out: dict[str, Any] = {}
    for i, msg in enumerate(messages or []):
        base = f"{prefix}.{i}.message"
        role = msg.get("role")
        if role:
            out[f"{base}.role"] = role
        content = msg.get("content")
        if content:
            out[f"{base}.content"] = content
        # Ollama puts the tool name on the tool result message, not an id.
        name = msg.get("tool_name")
        if name:
            out[f"{base}.name"] = name
        for j, call in enumerate(msg.get("tool_calls") or []):
            fn = call.get("function") or {}
            key = f"{base}.tool_calls.{j}.tool_call"
            if fn.get("name"):
                out[f"{key}.function.name"] = fn["name"]
            args = fn.get("arguments")
            if args is not None:
                out[f"{key}.function.arguments"] = (
                    args if isinstance(args, str) else json.dumps(args, default=str)
                )
    return out


def tool_attributes(tools: list[dict] | None) -> dict[str, Any]:
    """The tool catalogue offered to the model this round, as JSON schemas.

    Worth capturing per-round: Nova's router narrows the catalogue per turn, so
    "which tools could it even see?" is the first question when a model picks the
    wrong one - or picks nothing.
    """
    return {
        f"{LLM_TOOLS}.{i}.tool.json_schema": json.dumps(spec, default=str)
        for i, spec in enumerate(tools or [])
    }


def document_attributes(docs: list[dict] | None) -> dict[str, Any]:
    """Flatten retrieved passages into OpenInference's indexed document attributes.

    Same positional-dotted convention as messages. Recording the per-passage
    ``score`` is the point: when a manuals answer is wrong, the first question is
    whether retrieval surfaced the right pages or the model ignored good ones.
    """
    out: dict[str, Any] = {}
    for i, doc in enumerate(docs or []):
        base = f"{RETRIEVAL_DOCUMENTS}.{i}.document"
        if doc.get("id") is not None:
            out[f"{base}.id"] = str(doc["id"])
        if doc.get("content"):
            out[f"{base}.content"] = doc["content"]
        if doc.get("score") is not None:
            out[f"{base}.score"] = doc["score"]
        meta = doc.get("metadata")
        if meta:
            out[f"{base}.metadata"] = json.dumps(meta, default=str)
    return out


@contextlib.contextmanager
def activate(handle):
    """Make ``handle`` the *current* OTel span for the enclosed block.

    Needed only where third-party auto-instrumentation runs. Those libraries
    parent their spans off the implicit context, which this module otherwise
    avoids (see the streaming problem in the module docstring) - so without this
    an auto-instrumented call becomes a root span instead of a child of the turn.
    """
    raw = getattr(handle, "raw", None)
    if not _live or raw is None:
        yield
        return
    # Only the setup is guarded - the body's own exceptions must propagate
    # normally through the `with`, not be caught and re-yielded here.
    try:
        from opentelemetry import trace as _otel_trace

        scope = _otel_trace.use_span(raw, end_on_exit=False)
    except Exception:  # noqa: BLE001 - never let tracing break the call it wraps
        yield
        return
    with scope:
        yield


def _value_and_mime(value: Any) -> tuple[str, str]:
    """Render a value for `input.value` / `output.value`, which must be strings.

    A bare string is passed through as text; anything structured is JSON-encoded
    and declared as such, so the UI can pretty-print it instead of showing a blank.
    """
    if isinstance(value, str):
        return value, MIME_TEXT
    if isinstance(value, (int, float, bool)):
        return str(value), MIME_TEXT
    try:
        return json.dumps(value, default=str), MIME_JSON
    except (TypeError, ValueError):
        return str(value), MIME_TEXT


class _NoopSpan:
    """Stand-in with the real span's API, so callers never branch on enabled()."""

    __slots__ = ()

    raw = None

    def set(self, **attributes: Any) -> "_NoopSpan":
        return self

    def set_input(self, value: Any) -> "_NoopSpan":
        return self

    def set_output(self, value: Any) -> "_NoopSpan":
        return self

    def add_metadata(self, fields: dict) -> "_NoopSpan":
        return self

    def record_error(self, exc: BaseException) -> "_NoopSpan":
        return self

    def end(self) -> None:
        return None


NOOP = _NoopSpan()


class _Span:
    """Thin wrapper over an OTel span: coerces attributes and never raises."""

    __slots__ = ("_span", "_meta")

    def __init__(self, span) -> None:
        self._span = span
        self._meta: dict[str, Any] = {}

    @property
    def raw(self):
        """The underlying OTel span - pass as ``parent=`` to nest a child."""
        return self._span

    def set(self, **attributes: Any) -> "_Span":
        for key, value in attributes.items():
            if value is None:
                continue
            try:
                # Callers pass the dotted convention names via **{CONST: v},
                # so keys arrive already correct - no mangling here.
                self._span.set_attribute(key, _coerce(value))
            except Exception:  # noqa: BLE001
                pass
        return self

    def set_input(self, value: Any) -> "_Span":
        text, mime = _value_and_mime(value)
        return self.set(**{INPUT_VALUE: text, INPUT_MIME: mime})

    def set_output(self, value: Any) -> "_Span":
        text, mime = _value_and_mime(value)
        return self.set(**{OUTPUT_VALUE: text, OUTPUT_MIME: mime})

    def add_metadata(self, fields: dict) -> "_Span":
        """Merge into `metadata` rather than replacing it.

        `metadata` is a single JSON-encoded attribute, so a second plain
        ``set(metadata=...)`` silently overwrites the first. The turn span writes
        it twice - once when the escalation decision is made, once when the stream
        ends - which quietly discarded source, mode, skills and the escalation
        flags: exactly the fields an eval needs to slice on. Accumulate instead.
        """
        if not fields:
            return self
        self._meta.update(fields)
        return self.set(**{METADATA: dict(self._meta)})

    def record_error(self, exc: BaseException) -> "_Span":
        try:
            from opentelemetry.trace import Status, StatusCode

            self._span.record_exception(exc)
            self._span.set_status(Status(StatusCode.ERROR, str(exc)))
        except Exception:  # noqa: BLE001
            pass
        return self

    def end(self) -> None:
        try:
            self._span.end()
        except Exception:  # noqa: BLE001
            pass


def start(name: str, kind: str | None = None, parent=None, **attributes: Any):
    """Start a span the caller owns and must ``end()`` itself.

    Used where a ``with`` block can't reach - most importantly the SSE generator
    in ``/api/chat``, whose turn span has to stay open across every ``yield``.
    Pass ``parent=other.raw`` to nest.
    """
    if not _live or _tracer is None:
        return NOOP
    try:
        from opentelemetry import trace as _otel_trace

        ctx = _otel_trace.set_span_in_context(parent) if parent is not None else None
        span = _Span(_tracer.start_span(name, context=ctx))
    except Exception as exc:  # noqa: BLE001
        log.debug("arize: could not start span %r: %s", name, exc)
        return NOOP
    if kind:
        span.set(**{SPAN_KIND: kind})
    if attributes:
        span.set(**attributes)
    return span


@contextlib.contextmanager
def span(name: str, kind: str | None = None, parent=None, **attributes: Any):
    """Scoped span for ordinary (non-streaming) code paths."""
    handle = start(name, kind=kind, parent=parent, **attributes)
    try:
        yield handle
    except BaseException as exc:
        handle.record_error(exc)
        handle.end()
        raise
    handle.end()
