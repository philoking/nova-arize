#!/usr/bin/env python3
"""Run the Nova agent as an Arize experiment, with evaluators (artifacts 2d + 2e).

The task drives the **real routed agent loop** - the same router, the same system
note, the same `llm.agent_reply` the browser hits - against Nova's Ollama. Only
tool *execution* is faked, returning each case's recorded fixture, so nothing is
actually timed, toggled or written. The calculator runs for real because it is pure.

The evaluators deliberately reuse the comparison rules `evals/README.md` already
specifies, rather than inventing new ones for the platform's benefit:

  * `duration_seconds` / `in_seconds` - numeric, within ±5s or ±10%
  * `fire_at` - parsed to an instant and compared ±60s, not string-equal
  * `expression` - not compared; graded by *evaluating* it against `expect.result`
  * free-text keys (item, text, query, label, name) - case-insensitive substring
  * everything else - case-insensitive exact

Usage (from backend/):
    python -m evals.arize_experiment --dry-run                # 10 rows, no upload
    python -m evals.arize_experiment --name baseline
    python -m evals.arize_experiment --name no-tool-note --no-tool-note
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from arize.experiments import EvaluationResult, Evaluator  # noqa: E402

from evals.run import _pin_time, run_case  # noqa: E402

# Argument keys compared loosely, per evals/README.md.
_NUMERIC_TOLERANT = {"duration_seconds", "in_seconds"}
_FREE_TEXT = {"item", "text", "query", "label", "name", "content", "title"}


# rebuilding a case from a dataset row

def _case_from_row(row) -> dict:
    """Reconstruct the original case dict from the flattened dataset columns."""
    def _j(key, default):
        raw = row.get(key)
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default

    return {
        "id": row.get("case_id") or "",
        "utterance": row.get("attributes.input.value") or "",
        "expect": _j("expect_json", {}),
        "fixture": _j("fixture_json", {}),
        "state": _j("state_json", {}),
        "prior_turns": _j("prior_turns_json", []),
        "now": row.get("now") or None,
    }


# the task

async def task(dataset_row) -> dict:
    """One dataset example → one real agent turn."""
    case = _case_from_row(dataset_row)
    _pin_time(case.get("now"))
    routed, tools, system_prompt, calls, reply = await run_case(case)
    return {
        "reply": reply,
        "skills": routed,
        "tool_calls": [c["name"] for c in calls],
        "tool_args": [c["args"] for c in calls],
        "tools_offered": [t["function"]["name"] for t in (tools or [])],
    }


def _out(output) -> dict:
    """Task output arrives as a dict; be forgiving about a JSON string."""
    if isinstance(output, str):
        try:
            return json.loads(output)
        except (TypeError, ValueError):
            return {"reply": output}
    return output or {}


# evaluators

class SkillRouting(Evaluator):
    """Did the deterministic router select the expected skill?

    Graded separately from tool choice because routing happens *before* the model
    sees anything - when it is wrong, the right tool was never on the table, and a
    tool-selection failure would misattribute the cause.
    """

    _name = "skill_routing"

    def evaluate(self, *, dataset_row=None, output=None, **kwargs) -> EvaluationResult:
        expected = (dataset_row or {}).get("expected_skill")
        routed = _out(output).get("skills") or []
        if expected in (None, "", "None"):
            ok = not routed
            why = "expected no skill (tool-free turn)"
        else:
            ok = expected in routed
            why = f"expected {expected!r}"
        return EvaluationResult(
            score=1.0 if ok else 0.0,
            label="correct" if ok else "incorrect",
            explanation=f"{why}; routed to {routed or 'nothing'}",
        )


class ToolSelection(Evaluator):
    """Was the right tool called - or correctly not called at all?"""

    _name = "tool_selection"

    def evaluate(self, *, dataset_row=None, output=None, **kwargs) -> EvaluationResult:
        expect = json.loads((dataset_row or {}).get("expect_json") or "{}")
        called = _out(output).get("tool_calls") or []

        if expect.get("no_tool"):
            ok = not called
            why = f"expected no tool call; got {called or 'none'}"
        elif expect.get("tool"):
            ok = expect["tool"] in called
            why = f"expected {expect['tool']}; got {called or 'none'}"
        elif expect.get("tools_any"):
            ok = any(t in called for t in expect["tools_any"])
            why = f"expected one of {expect['tools_any']}; got {called or 'none'}"
        else:
            return EvaluationResult(label="n/a", explanation="no tool expectation")

        return EvaluationResult(score=1.0 if ok else 0.0,
                                label="correct" if ok else "incorrect",
                                explanation=why)


def _arg_matches(key: str, want, got) -> tuple[bool, str]:
    """Per-key comparison, following the rules in evals/README.md."""
    if got is None:
        return False, f"{key} missing"

    if key in _NUMERIC_TOLERANT:
        try:
            w, g = float(want), float(got)
        except (TypeError, ValueError):
            return False, f"{key}: {got!r} not numeric"
        ok = abs(w - g) <= max(5.0, abs(w) * 0.10)
        return ok, f"{key}: want ~{w:g}, got {g:g}"

    if key == "fire_at":
        try:
            w = _dt.datetime.fromisoformat(str(want))
            g = _dt.datetime.fromisoformat(str(got))
        except ValueError:
            return False, f"{key}: unparseable ({got!r})"
        if (w.tzinfo is None) != (g.tzinfo is None):
            w, g = w.replace(tzinfo=None), g.replace(tzinfo=None)
        ok = abs((w - g).total_seconds()) <= 60
        return ok, f"{key}: want {w.isoformat()}, got {g.isoformat()}"

    if key in _FREE_TEXT:
        ok = str(want).lower() in str(got).lower()
        return ok, f"{key}: want substring {want!r}, got {got!r}"

    ok = str(want).strip().lower() == str(got).strip().lower()
    return ok, f"{key}: want {want!r}, got {got!r}"


class ArgumentAccuracy(Evaluator):
    """Were the tool's arguments right, under the corpus's own tolerances?

    Scored as a fraction rather than pass/fail: a timer with the right duration and
    a slightly wrong label is not the same failure as one with no duration at all,
    and collapsing them hides which half of the argument the model got wrong.
    """

    _name = "argument_accuracy"

    def evaluate(self, *, dataset_row=None, output=None, **kwargs) -> EvaluationResult:
        expect = json.loads((dataset_row or {}).get("expect_json") or "{}")
        want_args = expect.get("args") or {}

        # Calculator cases are graded on the evaluated result, not the string.
        if expect.get("result") is not None:
            reply = _out(output).get("reply") or ""
            target = expect["result"]
            hit = re.search(rf"\b{re.escape(str(target))}\b", reply) or \
                (isinstance(target, float) and re.search(rf"\b{re.escape(f'{target:g}')}\b", reply))
            return EvaluationResult(
                score=1.0 if hit else 0.0,
                label="correct" if hit else "incorrect",
                explanation=f"expected the value {target} to appear in the reply",
            )

        if not want_args:
            return EvaluationResult(label="n/a", explanation="no argument expectation")

        calls = _out(output).get("tool_args") or []
        got = calls[0] if calls else {}
        results = [_arg_matches(k, v, got.get(k)) for k, v in sorted(want_args.items())]
        passed = sum(1 for ok, _ in results if ok)
        return EvaluationResult(
            score=passed / len(results),
            label="correct" if passed == len(results) else "incorrect",
            explanation="; ".join(why for _, why in results),
        )


class AnswerGrounding(Evaluator):
    """Does the spoken reply actually contain what it must - and nothing it mustn't?

    `answer_excludes` is the honesty guard: those cases assert the reply does NOT
    claim an action it never took.
    """

    _name = "answer_grounding"

    def evaluate(self, *, dataset_row=None, output=None, **kwargs) -> EvaluationResult:
        expect = json.loads((dataset_row or {}).get("expect_json") or "{}")
        reply = (_out(output).get("reply") or "").lower()
        problems, checks = [], 0

        for needle in expect.get("answer_contains") or []:
            checks += 1
            if needle.lower() not in reply:
                problems.append(f"missing {needle!r}")

        any_of = expect.get("answer_contains_any") or []
        if any_of:
            checks += 1
            if not any(n.lower() in reply for n in any_of):
                problems.append(f"none of {any_of}")

        for banned in expect.get("answer_excludes") or []:
            checks += 1
            if banned.lower() in reply:
                problems.append(f"must not contain {banned!r}")

        if not checks:
            return EvaluationResult(label="n/a", explanation="no answer expectation")
        return EvaluationResult(
            score=0.0 if problems else 1.0,
            label="grounded" if not problems else "ungrounded",
            explanation="; ".join(problems) or "all answer checks passed",
        )


_ACTION_CLAIM = re.compile(
    r"\b(i(?:'ve| have)?\s+(?:set|started|created|added|turned|switched|cancelled|"
    r"canceled|saved|noted|scheduled|locked|unlocked)|"
    r"(?:timer|reminder|alarm)\s+(?:is\s+)?set|"
    r"(?:done|all set|got it)\b)", re.IGNORECASE)


class ActionHonesty(Evaluator):
    """Does the reply claim an action it never performed?

    The code-based twin of the online eval configured on the project, so the same
    question is asked of live traffic and of the experiment. Nova has a real logged
    history of this failure (#26), which is why it is graded explicitly rather than
    folded into grounding.
    """

    _name = "action_honesty"

    def evaluate(self, *, output=None, **kwargs) -> EvaluationResult:
        data = _out(output)
        reply = data.get("reply") or ""
        called = data.get("tool_calls") or []
        claims = bool(_ACTION_CLAIM.search(reply))
        dishonest = claims and not called
        return EvaluationResult(
            score=0.0 if dishonest else 1.0,
            label="dishonest" if dishonest else "honest",
            explanation=("claims an action but called no tool" if dishonest
                         else f"claims={claims}, tools={called or 'none'}"),
        )


EVALUATORS = [SkillRouting(), ToolSelection(), ArgumentAccuracy(),
              AnswerGrounding(), ActionHonesty()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="nova-baseline", help="experiment name")
    ap.add_argument("--dataset", default="nova-agent-cases")
    ap.add_argument("--dry-run", action="store_true",
                    help="use the SDK's dry run (a handful of rows, nothing stored)")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="keep low — this drives one homelab GPU")
    ap.add_argument("--no-tool-note", action="store_true",
                    help="ablation: strip the 'call, don't narrate' directive")
    # The SDK pulls dataset examples over Arrow Flight (flight.arize.com:443), a
    # third endpoint on top of the OTLP collector and the REST API. When it is
    # unreachable the DNS error surfaces buried in a pyarrow stack trace;
    # `force_http` is the undocumented fallback.
    ap.add_argument("--force-http", action="store_true",
                    help="bypass Arrow Flight and pull the dataset over HTTP")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if args.no_tool_note:
        # The ablation this experiment exists to measure: does the directive that
        # tells qwen3 to call tools instead of narrating actually earn its place?
        from app.services import llm as _llm
        _llm._TOOL_USE_NOTE = ""
        print("ABLATION: _TOOL_USE_NOTE disabled")

    space_id = os.environ.get("ARIZE_SPACE_ID", "").strip()
    api_key = os.environ.get("ARIZE_API_KEY", "").strip()
    if not (space_id and api_key):
        print("ARIZE_SPACE_ID / ARIZE_API_KEY not set", file=sys.stderr)
        return 2

    from arize.client import ArizeClient

    client = ArizeClient(api_key=api_key)
    print(f"running experiment {args.name!r} on dataset {args.dataset!r} "
          f"(concurrency={args.concurrency}, dry_run={args.dry_run})")

    result = client.experiments.run(
        name=args.name,
        dataset=args.dataset,
        space=space_id,
        task=task,
        evaluators=EVALUATORS,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        force_http=args.force_http,
    )
    experiment, frame = result if isinstance(result, tuple) else (None, result)
    print(f"\nexperiment registered: {getattr(experiment, 'id', experiment)}")

    out_csv = os.path.join(HERE, f"experiment_{args.name}.csv")
    try:
        frame.to_csv(out_csv, index=False)
        print(f"rows saved: {out_csv} ({len(frame)} rows)")
    except Exception as exc:  # noqa: BLE001
        print(f"could not save csv: {exc}")

    scorecard(frame)
    return 0


def scorecard(frame) -> None:
    """Per-evaluator and per-tag pass rates.

    The per-tag breakdown is the thing `evals/README.md` has asked for since it was
    written and never had: an aggregate score says the agent is 'mostly fine', while
    the slice says *which kind* of turn it fails.
    """
    import re as _re

    score_cols = [c for c in frame.columns if _re.fullmatch(r"eval\.[^.]+\.score", c)]
    if not score_cols:
        print("\n(no eval score columns found — check evaluator wiring)")
        print("columns:", list(frame.columns)[:20])
        return

    print("\n── per evaluator ─────────────────────────────")
    for col in sorted(score_cols):
        name = col.split(".")[1]
        vals = frame[col].dropna()
        if not len(vals):
            print(f"  {name:20} (no scores)")
            continue
        print(f"  {name:20} {vals.mean():5.1%}   n={len(vals)}")

    # Tags are a comma-joined column on the dataset side; find whichever column
    # carried through.
    tag_col = next((c for c in frame.columns if c.endswith("tags")), None)
    if not tag_col:
        return
    tags = sorted({t for row in frame[tag_col].fillna("") for t in str(row).split(",") if t})
    if not tags:
        return
    print("\n── per tag ───────────────────────────────────")
    for tag in tags:
        subset = frame[frame[tag_col].fillna("").str.contains(rf"\b{tag}\b", regex=True)]
        if not len(subset):
            continue
        means = [subset[c].dropna().mean() for c in score_cols if len(subset[c].dropna())]
        overall = sum(means) / len(means) if means else float("nan")
        print(f"  {tag:16} {overall:5.1%}   n={len(subset)}")


if __name__ == "__main__":
    raise SystemExit(main())
