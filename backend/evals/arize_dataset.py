#!/usr/bin/env python3
"""Upload the Nova eval corpus to Arize as a dataset (take-home artifact 2c).

`cases.jsonl` is 55 hand-written cases: an utterance plus what the agent *should*
do with it - which skill it routes to, which tool it calls, with what arguments,
and whether the spoken reply is grounded in the tool result. It predates this
exercise and was written to test the agent, not to please a platform, which makes
it a fair test of whether Arize's dataset model fits data it did not shape.

The awkward part, worth noting up front: this corpus has **no reference answer**.
The expectation is structured - `timers__start_timer(duration_seconds=600)` - not
a golden string. Arize's dataset shape wants `attributes.output.value`, so the
structured expectation is carried in metadata columns for evaluators to read, and
`output.value` gets a human-readable rendering purely so the UI shows something
meaningful when you eyeball a row.

Usage (from backend/):
    python -m evals.arize_dataset --dry-run          # print the frame, no upload
    python -m evals.arize_dataset --name nova-agent-v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

CASES = os.path.join(HERE, "cases.jsonl")


def load_cases() -> list[dict]:
    with open(CASES, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def expectation_text(expect: dict) -> str:
    """A readable one-line rendering of what the agent should do.

    Only for human eyes in the dataset UI - every evaluator reads the structured
    `expect_json` column instead, so nothing depends on this format.
    """
    if expect.get("no_tool"):
        return "(no tool call)"
    tool = expect.get("tool") or "|".join(expect.get("tools_any") or []) or "(none)"
    args = expect.get("args") or {}
    if args:
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(args.items()))
        return f"{tool}({rendered})"
    if expect.get("result") is not None:
        return f"{tool} → {expect['result']}"
    return tool


def to_frame(cases: list[dict]):
    import pandas as pd

    rows = []
    for case in cases:
        expect = case.get("expect") or {}
        rows.append({
            # Arize's expected input/output columns.
            "attributes.input.value": case["utterance"],
            "attributes.output.value": expectation_text(expect),
            # Structured expectation - what evaluators actually grade against.
            "case_id": case["id"],
            "expect_json": json.dumps(expect, sort_keys=True),
            "expected_skill": str(expect.get("skill")),
            "expected_tool": expect.get("tool") or "",
            "expects_no_tool": bool(expect.get("no_tool")),
            # Slices. `tags` is a list; joined because a dataset column has to be
            # scalar, and re-split by anything that filters on it.
            "tags": ",".join(case.get("tags") or []),
            # World the case runs in - the harness needs these to reproduce it.
            "fixture_json": json.dumps(case.get("fixture") or {}, sort_keys=True),
            "state_json": json.dumps(case.get("state") or {}, sort_keys=True),
            "prior_turns_json": json.dumps(case.get("prior_turns") or [], sort_keys=True),
            # Pinned clock for time-relative cases ("remind me at 5pm").
            "now": case.get("now") or "",
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="nova-agent-cases",
                    help="dataset name in Arize")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the frame without uploading")
    ap.add_argument("--limit", type=int, help="only the first N cases")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]
    frame = to_frame(cases)

    print(f"{len(frame)} cases, {len(frame.columns)} columns")
    print(f"columns: {list(frame.columns)}\n")
    tagged = (frame["tags"] != "").sum()
    print(f"tagged: {tagged}/{len(frame)}   with a tool: {(frame['expected_tool'] != '').sum()}"
          f"   tool-free: {frame['expects_no_tool'].sum()}\n")
    with_pd_options(frame)

    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    space_id = os.environ.get("ARIZE_SPACE_ID", "").strip()
    api_key = os.environ.get("ARIZE_API_KEY", "").strip()
    if not (space_id and api_key):
        print("ARIZE_SPACE_ID / ARIZE_API_KEY not set", file=sys.stderr)
        return 2

    # The published quickstart imports `arize.experimental.datasets`, which does
    # not exist in arize 8.51.0 - no shim, no pointer to the replacement. Datasets,
    # experiments and evaluators are top-level sub-clients on ArizeClient, with
    # different method and parameter names.
    from arize.client import ArizeClient

    client = ArizeClient(api_key=api_key)
    dataset = client.datasets.create(
        name=args.name,
        space=space_id,
        examples=frame,
    )
    print(f"\ncreated dataset {args.name!r} -> {getattr(dataset, 'id', dataset)}")
    return 0


def with_pd_options(frame) -> None:
    import pandas as pd

    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.max_colwidth", 40):
        print(frame[["case_id", "attributes.input.value",
                     "attributes.output.value", "tags"]].head(12).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
