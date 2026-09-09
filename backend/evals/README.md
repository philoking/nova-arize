# Agent eval corpus (#60)

`cases.jsonl` is the seed eval set for the **tool-calling agent** — the layer above
the router (`test_skills.py`) and the tools (`notes_harness.py`, #27). Each line is
one case: a natural-language utterance and what the agent *should* do with it —
which skill it routes to, which tool it calls, with what arguments, and whether the
final spoken reply is grounded in the tool result (and doesn't fake an action).

The harness that consumes this file doesn't exist yet — this is the data first. See
#60 for the harness plan. The schema below is what the harness is expected to read.

## How a case is graded

Drive the real routed loop for the utterance:

```
skills = registry.route(user_texts)          # deterministic router
tools  = registry.tools_for(skills)
note   = registry.context_for(skills)
stream = llm.agent_reply(history, tools, fake_execute_tool, note, response_mode="voice")
```

`fake_execute_tool` **records** every `(tool, args)` call and returns the case's
`fixture` (so nothing real is timed/toggled/written). The calculator and the router
are pure, so those cases MAY use the real executor instead of a fixture.

Then score the recorded calls + final text against `expect`.

## Case schema

| field | type | meaning |
|---|---|---|
| `id` | string | stable id, `<skill>-NN`, for the per-case scorecard |
| `utterance` | string | the user's message for this turn |
| `prior_turns` | array | optional `{role, content}` history before the utterance (follow-ups); only the `user` items feed the router |
| `now` | string | optional pinned local time (ISO-8601). The harness freezes `timers._now()`/`datetime.now` to this so time-relative args are deterministic |
| `state` | object | optional world the system note reflects — `ha_devices` (list; absent ⇒ HA disabled ⇒ the `home` skill isn't offered), `ringing` (list of ringing alarms), `memories` (durable facts) |
| `fixture` | object | what `fake_execute_tool` returns for the expected tool call (shapes mirror the real providers) |
| `tags` | array | slice labels for the scorecard: `safety`, `honesty`, `grounding`, `followup`, `hygiene`, `negative`, `hard`, `clarify`, `personalization` |
| `expect` | object | the assertions, below |

### `expect`

| field | type | check |
|---|---|---|
| `skill` | string \| null | the routed skill set **must contain** this skill; `null` ⇒ the set must be **empty** (tool-free turn) |
| `tool` | string | the namespaced tool (`provider__tool`) the first tool call must be |
| `tools_any` | array | acceptable alternatives when more than one tool is defensible (multi-round / find-first flows) |
| `no_tool` | bool | the turn must make **zero** tool calls |
| `args` | object | expected arguments on the tool call. Per-key comparison rules below |
| `result` | number | calculator only: the value `args.expression` must **evaluate to** (the string itself is not matched — `240*0.15`, `0.15*240`, `240*15/100` all pass) |
| `answer_contains` | array | **all** substrings must appear in the final reply (case-insensitive) — use for a required value like `"36"` |
| `answer_contains_any` | array | **at least one** must appear — use for natural-language variants |
| `answer_excludes` | array | none may appear — used for the honesty guard (must not claim it acted) |

**Per-key `args` comparison**

- `duration_seconds`, `in_seconds` — numeric, within ±5 s or ±10 %.
- `fire_at` — parsed to an instant (via the pinned `now`) and compared ±60 s, not string-equal.
- `expression` — not compared; graded via `expect.result` (see above).
- free-text keys (`item`, `text`, `query`, `label`, `name`) — case-insensitive **substring** match.
- everything else (`section`, `domain`, `service`, `entity_id`, `when`, `horizon`, `path`, `project`, `title`) — case-insensitive exact match.

## Global invariants (asserted on every case, not per-line)

- No `<think>`/`</think>` (or other qwen3 control tokens) ever appear in the reply (#19).
- If `expect.tool`/`tools_any` is set, a real tool call must occur — a "let me check…"
  narration with no call is a fail (the `agent_reply` nudge exists for this).
- The reply never claims a device/timer/note action unless a tool actually ran and
  succeeded this turn (#26). Cases that specifically probe this carry `answer_excludes`.

## Determinism

`_stream_once` uses `temperature: 0.7`. The harness should run each case at temp 0
and/or pass@k with a threshold, and emit a **pass-rate per tag per model** scorecard
(`qwen3:8b` vs `:14b` vs `qwen3.5:9b`) — a report, not a hard CI gate, at first.
