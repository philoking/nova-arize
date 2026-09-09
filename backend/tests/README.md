# Tests

Testing is in three tiers, by how much they need to run:

1. **Unit suite** (`tests/test_*.py`) — 87 fast, hermetic tests. No network, GPU,
   DB file, or broker. This is what you run before every push.
2. **Notes harness** (`tests/notes_harness.py`, #27) — drives the real Obsidian
   vault; **not** auto-discovered, run deliberately.
3. **Agent evals** (`../evals/`, #60) — model-in-the-loop against Nova's Ollama;
   separate runner, see [`../evals/README.md`](../evals/README.md).

## Running the unit suite

From `backend/`:

```bash
python -m unittest discover -s tests
```

Stdlib `unittest` only — no test dependency to install, runs in well under a
second. Files named `notes_harness.py` (no `test_` prefix) are ignored by
discovery, so the live-vault harness never runs by accident.

## What the unit suite covers

| File | Tests | Covers | Issue |
|---|---:|---|---|
| `test_skills.py` | 16 | Capability routing: deterministic keyword matching, follow-up (pronoun) inheritance, the math heuristic, and skill availability. Includes the query that regressed #26. | #35 / #39 |
| `test_calculator.py` | 15 | The safe AST evaluator (arithmetic correctness + rejection of anything non-arithmetic), `date_diff` calendar math, the provider `execute()` contract, and the math routing heuristic. | #54 |
| `test_recurrence.py` | 14 | Recurring-reminder scheduling: repeat normalization, next-occurrence, cadence labels, and alarms. Fixed timezone so day/time assertions are deterministic. | #15 |
| `test_memory.py` | 10 | The persistent-memory store (add / list / tier / prune). Throwaway DB. | #29 |
| `test_manuals.py` | 11 | Shop-manual RAG: text chunking (whitespace/overlap/page tagging), title derivation, and the `search_manuals` provider contract with the embedding + Qdrant calls stubbed. No live services or PDF fixture. | #63 |
| `test_escalation.py` | 7 | Auto-escalation triggers — the reactive struggle detector (empty/punt vs. substantive) and the upfront intent detector (research/writing/"ask Claude" → escalate; simple turns stay local). Pure/deterministic. | #63 / #65 |
| `test_notify.py` | 9 | MQTT notification **gating** logic — global toggle, per-category allowlist, timer→category mapping. No broker; `aiomqtt` imported lazily so no dependency needed. | #16 |
| `test_registry_scrub.py` | 8 | The tool-argument scrubber that strips qwen3 `/no_think` and `<think>…</think>` control tokens out of string args before they reach a provider. | #19 |
| `test_lists.py` | 7 | List `remove_item` / `clear_list` against a **fake** Obsidian REST client, including the ambiguity guard (asks instead of guessing). | #14 |
| `test_sessions.py` | 4 | Server-owned sessions: `history.thread_for` rehydration and cross-surface continuity (a thread started on the satellite continues from the web with full context). Throwaway DB. | #44 |
| `test_logstore.py` | 4 | The pure activity-log classifier — routine polling (hideable) vs. meaningful activity. | #55 |

## Conventions

- **Hermetic.** Every test avoids the network, the GPU, and real service state.
  External services are exercised through **fakes** (e.g. `test_lists` injects a
  fake Obsidian client) or not at all (only pure decision logic is tested, e.g.
  `test_notify` covers gating without a broker).
- **Throwaway state.** Tests that touch SQLite / settings point at a temp file via
  an env var set **before** importing the app (see `test_memory`, `test_sessions`,
  `test_notify`) so they never touch your real `data/`.
- **Determinism.** Time-sensitive tests pin a fixed timezone (`test_recurrence`)
  rather than depending on the host zone.
- Prefer testing the **pure** seam. Most bugs here are in decision logic (routing,
  gating, scrubbing, evaluation), which is extracted so it's testable without the
  model or a live service.

## Deliberately out of scope for the unit suite

- **Live model behaviour** (does qwen3 pick the right tool, fill args, stay
  grounded) → the agent evals in [`../evals/`](../evals/README.md) (#60).
- **Real service integration** (STT / TTS / HA / Obsidian over the network) →
  covered by fakes here; the notes tool path is exercised end-to-end by
  `notes_harness.py` (#27) against the real vault, safety-gated by a `nova-test-`
  prefix.
- **Frontend JS** — no automated coverage; verified by hand / screenshot.

## The notes harness (manual)

`notes_harness.py` (#27) runs the Obsidian `notes` tools directly (no LLM) against
the **real** vault to check what actually lands, phrasing by phrasing. It is safe
because everything it creates is prefixed `nova-test-` and destructive ops only
ever target exact `nova-test-` paths; it cleans up after itself. Run it deliberately
(it needs the vault reachable):

```bash
python -m tests.notes_harness      # from backend/
```

## CI

The Gitea Actions deploy workflow (`.gitea/workflows/deploy.yml`) does **not** run
the suite — it builds, deploys, and health-checks. So the unit suite is a **local
pre-push gate**: run `python -m unittest discover -s tests` before pushing to
`main`. (Adding a test job to CI is a reasonable future step — see the M5 · Ops
milestone.)
