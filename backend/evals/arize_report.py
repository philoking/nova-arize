#!/usr/bin/env python3
"""Per-tag scorecard and failure list for an Arize experiment run.

The frame `experiments.run` returns carries only `example_id`, `output`, `error`
and the eval columns; every column the dataset carried (`case_id`, `tags`,
`expect_json`, even the input text) is dropped. Slicing by those tags means
fetching the examples separately, unwrapping a nested `additional_properties`,
and re-joining on an opaque UUID. Tagging cases `safety` / `honesty` /
`grounding` is the point of the corpus, and none of it survives without this.

Usage (needs ARIZE_SPACE_ID / ARIZE_API_KEY):
    python -m evals.arize_report --csv evals/experiment_nova-baseline-v2.csv
    python -m evals.arize_report --csv a.csv --compare b.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SCORE_RE = re.compile(r"eval\.([^.]+)\.score")


def load_examples(dataset: str):
    """Dataset rows, with the payload lifted out of `additional_properties`."""
    import pandas as pd
    from arize.client import ArizeClient

    client = ArizeClient(api_key=os.environ["ARIZE_API_KEY"])
    res = client.datasets.list_examples(
        dataset=dataset, space=os.environ["ARIZE_SPACE_ID"], all=True)
    rows = []
    for ex in (getattr(res, "examples", res) or []):
        payload = getattr(ex, "additional_properties", None) or {}
        rows.append({**payload, "example_id": getattr(ex, "id", None)})
    return pd.DataFrame(rows)


def joined(csv_path: str, dataset: str):
    import pandas as pd

    results = pd.read_csv(csv_path)
    return results.merge(load_examples(dataset), on="example_id", how="left")


def score_columns(frame) -> list[str]:
    return [c for c in frame.columns if SCORE_RE.fullmatch(c)]


def per_evaluator(frame) -> dict[str, float]:
    out = {}
    for col in sorted(score_columns(frame)):
        vals = frame[col].dropna()
        if len(vals):
            out[SCORE_RE.fullmatch(col).group(1)] = float(vals.mean())
    return out


def per_tag(frame) -> dict[str, tuple[float, int]]:
    cols = score_columns(frame)
    tags = sorted({t for v in frame["tags"].fillna("") for t in str(v).split(",") if t})
    out = {}
    for tag in tags:
        subset = frame[frame["tags"].fillna("").str.contains(tag, regex=False)]
        means = [subset[c].dropna().mean() for c in cols if len(subset[c].dropna())]
        if means:
            out[tag] = (sum(means) / len(means), len(subset))
    return out


def failures(frame) -> list[tuple[str, str, str, str]]:
    out = []
    for col in score_columns(frame):
        name = SCORE_RE.fullmatch(col).group(1)
        why_col = col.replace(".score", ".explanation")
        for _, row in frame[frame[col] < 1.0].iterrows():
            out.append((name, str(row.get("case_id")),
                        str(row.get("attributes.input.value"))[:46],
                        str(row.get(why_col))[:80]))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="experiment results CSV")
    ap.add_argument("--compare", help="a second run to diff against")
    ap.add_argument("--dataset", default="nova-agent-cases")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    frame = joined(args.csv, args.dataset)
    print(f"{len(frame)} rows from {os.path.basename(args.csv)}\n")

    print("== per evaluator ==")
    base = per_evaluator(frame)
    for name, score in base.items():
        print(f"  {name:20} {score:6.1%}")

    print("\n== per tag ==")
    for tag, (score, n) in per_tag(frame).items():
        print(f"  {tag:18} {score:6.1%}  n={n}")

    print("\n== failures ==")
    for name, case, utterance, why in failures(frame):
        print(f"  [{name}] {case:16} {utterance:48} {why}")

    if args.compare:
        other = per_evaluator(joined(args.compare, args.dataset))
        print(f"\n== vs {os.path.basename(args.compare)} ==")
        print("  NOTE: single run, temperature 0.7, n=55. A 2pp move is one case.")
        print("  Read the direction of agreement across metrics, not any single delta.\n")
        for name in sorted(set(base) | set(other)):
            a, b = base.get(name), other.get(name)
            if a is None or b is None:
                continue
            delta = (a - b) * 100
            mark = "  " if abs(delta) < 0.05 else ("↑ " if delta > 0 else "↓ ")
            print(f"  {mark}{name:20} {a:6.1%}  vs {b:6.1%}   {delta:+.1f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
