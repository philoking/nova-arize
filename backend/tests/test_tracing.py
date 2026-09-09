"""Unit tests for the Arize/OpenInference tracing layer.

The whole point of these is the *disabled* path: tracing is optional, the OTel
packages are not test dependencies, and the app must behave identically when it
is off. So these run with nothing installed and assert that every helper is a
harmless no-op which still honours the real API (chainable, context-managed,
non-swallowing). Stdlib only.

Run from ``backend/``:  python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import tracing  # noqa: E402
from app.config import Settings  # noqa: E402


class Gating(unittest.TestCase):
    """``arize_enabled`` needs BOTH the space id and the key - a half-configured
    deploy must stay off rather than fail at export time."""

    def _settings(self, **env):
        orig = {k: os.environ.get(k) for k in ("ARIZE_SPACE_ID", "ARIZE_API_KEY")}
        try:
            for key in orig:
                os.environ.pop(key, None)
            os.environ.update(env)
            return Settings()
        finally:
            for key, val in orig.items():
                os.environ.pop(key, None)
                if val is not None:
                    os.environ[key] = val

    def test_off_by_default(self):
        self.assertFalse(self._settings().arize_enabled)

    def test_needs_both_values(self):
        self.assertFalse(self._settings(ARIZE_SPACE_ID="space").arize_enabled)
        self.assertFalse(self._settings(ARIZE_API_KEY="key").arize_enabled)

    def test_enabled_with_both(self):
        cfg = self._settings(ARIZE_SPACE_ID="space", ARIZE_API_KEY="key")
        self.assertTrue(cfg.arize_enabled)
        self.assertEqual(cfg.arize_project, "nova-voice")


class DisabledIsInert(unittest.TestCase):
    """With no exporter registered, every entry point returns the no-op and does
    nothing observable - no imports, no exceptions, no state."""

    def test_not_enabled(self):
        self.assertFalse(tracing.enabled())

    def test_init_returns_false_unconfigured(self):
        self.assertFalse(tracing.init())
        self.assertFalse(tracing.enabled())

    def test_start_returns_noop(self):
        span = tracing.start("chat.turn", kind=tracing.AGENT, **{tracing.INPUT_VALUE: "hi"})
        self.assertIs(span, tracing.NOOP)
        self.assertIsNone(span.raw)
        span.end()  # must not raise

    def test_noop_is_chainable(self):
        span = tracing.start("x")
        self.assertIs(span.set(**{tracing.SESSION_ID: "c1"}), span)
        self.assertIs(span.set_input("in"), span)
        self.assertIs(span.set_output("out"), span)
        self.assertIs(span.record_error(ValueError("boom")), span)

    def test_span_context_manager_yields_noop(self):
        with tracing.span("route", kind=tracing.CHAIN) as span:
            self.assertIs(span, tracing.NOOP)

    def test_span_does_not_swallow_exceptions(self):
        """A failing turn must still fail - the span wrapper records and re-raises."""
        with self.assertRaises(ValueError):
            with tracing.span("route", kind=tracing.CHAIN):
                raise ValueError("boom")

    def test_nested_parent_accepted_when_disabled(self):
        """``parent=span.raw`` is None while disabled; that must be fine."""
        root = tracing.start("chat.turn", kind=tracing.AGENT)
        with tracing.span("route", kind=tracing.CHAIN, parent=root.raw) as child:
            self.assertIs(child, tracing.NOOP)


class Coercion(unittest.TestCase):
    """OTel only accepts scalars and homogeneous scalar sequences, so richer
    values (tool args, routed skills, retrieval hits) are JSON-encoded."""

    def test_scalars_pass_through(self):
        for value in ("text", True, 3, 1.5):
            self.assertEqual(tracing._coerce(value), value)

    def test_string_list_preserved(self):
        self.assertEqual(tracing._coerce(["timers", "home"]), ["timers", "home"])

    def test_dict_becomes_json(self):
        self.assertEqual(tracing._coerce({"skills": ["timers"]}), '{"skills": ["timers"]}')

    def test_mixed_list_becomes_json(self):
        self.assertEqual(tracing._coerce([1, "a"]), '[1, "a"]')

    def test_unserialisable_falls_back_to_str(self):
        self.assertIn("object", tracing._coerce(object()))


class MessageFlattening(unittest.TestCase):
    """OpenInference's message attributes are positional and dotted, and nothing
    publishes a builder for them - so the shape is pinned here."""

    def test_role_and_content(self):
        attrs = tracing.message_attributes("llm.input_messages", [
            {"role": "system", "content": "you are nova"},
            {"role": "user", "content": "set a timer for 10 minutes"},
        ])
        self.assertEqual(attrs["llm.input_messages.0.message.role"], "system")
        self.assertEqual(attrs["llm.input_messages.1.message.content"],
                         "set a timer for 10 minutes")

    def test_empty_content_omitted(self):
        attrs = tracing.message_attributes("llm.input_messages", [{"role": "assistant", "content": ""}])
        self.assertNotIn("llm.input_messages.0.message.content", attrs)
        self.assertEqual(attrs["llm.input_messages.0.message.role"], "assistant")

    def test_tool_calls_flattened(self):
        attrs = tracing.message_attributes("llm.output_messages", [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "timers__start_timer",
                                         "arguments": {"duration_seconds": 600}}}],
        }])
        key = "llm.output_messages.0.message.tool_calls.0.tool_call.function"
        self.assertEqual(attrs[f"{key}.name"], "timers__start_timer")
        self.assertEqual(attrs[f"{key}.arguments"], '{"duration_seconds": 600}')

    def test_string_arguments_passed_through(self):
        """Ollama sometimes hands back arguments already JSON-encoded."""
        attrs = tracing.message_attributes("llm.output_messages", [{
            "role": "assistant",
            "tool_calls": [{"function": {"name": "t", "arguments": '{"a": 1}'}}],
        }])
        self.assertEqual(
            attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"],
            '{"a": 1}')

    def test_tool_result_name(self):
        attrs = tracing.message_attributes("llm.input_messages", [
            {"role": "tool", "tool_name": "timers__start_timer", "content": "{}"},
        ])
        self.assertEqual(attrs["llm.input_messages.0.message.name"], "timers__start_timer")

    def test_empty_history(self):
        self.assertEqual(tracing.message_attributes("llm.input_messages", []), {})
        self.assertEqual(tracing.message_attributes("llm.input_messages", None), {})

    def test_tool_schemas(self):
        attrs = tracing.tool_attributes([{"function": {"name": "calc"}}])
        self.assertIn('"calc"', attrs["llm.tools.0.tool.json_schema"])
        self.assertEqual(tracing.tool_attributes(None), {})


class DocumentFlattening(unittest.TestCase):
    """RETRIEVER spans carry the manuals hits. The per-passage score is the whole
    point: it separates 'retrieval missed' from 'the model ignored good passages'."""

    def test_full_document(self):
        attrs = tracing.document_attributes([{
            "id": "doc-1",
            "content": "Use a 3/8 inch blade.",
            "score": 0.71,
            "metadata": {"manual": "Grizzly G0555", "page": 14},
        }])
        base = "retrieval.documents.0.document"
        self.assertEqual(attrs[f"{base}.id"], "doc-1")
        self.assertEqual(attrs[f"{base}.content"], "Use a 3/8 inch blade.")
        self.assertEqual(attrs[f"{base}.score"], 0.71)
        self.assertIn("Grizzly", attrs[f"{base}.metadata"])

    def test_indexes_increment(self):
        attrs = tracing.document_attributes([{"id": "a"}, {"id": "b"}])
        self.assertEqual(attrs["retrieval.documents.0.document.id"], "a")
        self.assertEqual(attrs["retrieval.documents.1.document.id"], "b")

    def test_zero_score_is_kept(self):
        """0.0 is a real score - a falsy check here would silently drop it."""
        attrs = tracing.document_attributes([{"id": "a", "score": 0.0}])
        self.assertEqual(attrs["retrieval.documents.0.document.score"], 0.0)

    def test_missing_fields_omitted(self):
        attrs = tracing.document_attributes([{"id": "a"}])
        self.assertNotIn("retrieval.documents.0.document.content", attrs)
        self.assertNotIn("retrieval.documents.0.document.score", attrs)

    def test_empty(self):
        self.assertEqual(tracing.document_attributes([]), {})
        self.assertEqual(tracing.document_attributes(None), {})


class InputOutputValues(unittest.TestCase):
    """`input.value` / `output.value` are STRING attributes with a companion mime
    type. Sending a list produced an OTel array that Arize rendered as nothing -
    the `route` span appeared completely empty despite carrying both values."""

    def test_string_stays_text(self):
        self.assertEqual(tracing._value_and_mime("hello"), ("hello", "text/plain"))

    def test_list_becomes_json(self):
        text, mime = tracing._value_and_mime(["manuals", "web"])
        self.assertEqual(text, '["manuals", "web"]')
        self.assertEqual(mime, "application/json")

    def test_empty_list_is_still_a_string(self):
        """The routed-nothing case - must not vanish from the UI."""
        text, mime = tracing._value_and_mime([])
        self.assertEqual(text, "[]")
        self.assertEqual(mime, "application/json")

    def test_dict_becomes_json(self):
        text, mime = tracing._value_and_mime({"query": "blade"})
        self.assertEqual(text, '{"query": "blade"}')
        self.assertEqual(mime, "application/json")

    def test_scalars_are_text(self):
        self.assertEqual(tracing._value_and_mime(42), ("42", "text/plain"))
        self.assertEqual(tracing._value_and_mime(True), ("True", "text/plain"))

    def test_unknown_object_is_stringified_into_json(self):
        """`default=str` rescues most objects, so they become a JSON string rather
        than hitting the text fallback. Still a string attribute either way."""
        text, mime = tracing._value_and_mime(object())
        self.assertEqual(mime, "application/json")
        self.assertIn("object", text)

    def test_true_fallback_when_json_raises(self):
        """`default=` handles unserialisable *values*, not keys - a non-string key
        still raises, and that is the case the text fallback exists for."""
        text, mime = tracing._value_and_mime({(1, 2): "a"})
        self.assertEqual(mime, "text/plain")
        self.assertIsInstance(text, str)

    def test_mime_constants(self):
        self.assertEqual(tracing.INPUT_MIME, "input.mime_type")
        self.assertEqual(tracing.OUTPUT_MIME, "output.mime_type")


class Activate(unittest.TestCase):
    """``activate`` bridges to auto-instrumented libraries, which parent off the
    implicit OTel context. It must be transparent when tracing is off - and must
    never swallow the body's exceptions."""

    def test_noop_when_disabled(self):
        ran = False
        with tracing.activate(tracing.NOOP):
            ran = True
        self.assertTrue(ran)

    def test_accepts_bare_none(self):
        with tracing.activate(None):
            pass

    def test_propagates_exceptions(self):
        with self.assertRaises(ValueError):
            with tracing.activate(tracing.NOOP):
                raise ValueError("boom")


class Conventions(unittest.TestCase):
    """Guard the OpenInference attribute names. They are written as literals
    here (the manual-instrumentation docs never list them), so a typo would be
    silent in the UI - these pin the spelling."""

    def test_span_kinds(self):
        self.assertEqual(
            (tracing.AGENT, tracing.CHAIN, tracing.LLM, tracing.TOOL, tracing.RETRIEVER),
            ("AGENT", "CHAIN", "LLM", "TOOL", "RETRIEVER"),
        )

    def test_attribute_names(self):
        self.assertEqual(tracing.SPAN_KIND, "openinference.span.kind")
        self.assertEqual(tracing.INPUT_VALUE, "input.value")
        self.assertEqual(tracing.OUTPUT_VALUE, "output.value")
        self.assertEqual(tracing.SESSION_ID, "session.id")
        self.assertEqual(tracing.LLM_TOKEN_PROMPT, "llm.token_count.prompt")
        self.assertEqual(tracing.TOOL_NAME, "tool.name")


if __name__ == "__main__":
    unittest.main()


class MetadataMerging(unittest.TestCase):
    """`metadata` is one JSON attribute, so a second plain set() replaces it.

    The turn span writes metadata twice - once at the escalation decision, once
    when the stream ends - which silently discarded source, mode, skills and the
    escalation flags: exactly the fields an eval slices on.
    """

    def test_noop_accepts_add_metadata(self):
        span = tracing.start("chat.turn")
        self.assertIs(span.add_metadata({"a": 1}), span)

    def test_noop_handles_empty(self):
        self.assertIs(tracing.NOOP.add_metadata({}), tracing.NOOP)
