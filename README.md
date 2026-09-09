# Instrumenting a real agent with Arize AX

Nova Voice is a self-hosted voice assistant that has run in my house for
months: a local model on a homelab GPU driving eight tool providers, RAG over my
own equipment manuals, and an opt-in escalation to Claude on hard turns. This repo
is that application, plus the work of putting it under Arize AX
observability and evaluation.

I instrumented the agent I actually use rather than standing something up for the
exercise, on the grounds that a purpose-built demo would have hit tidier problems
than a running system does.

## What the agent is

A deterministic keyword router picks one or two skills per turn, then a local
model drives whatever tools those skills expose.

- Local model `qwen3:8b` on a homelab RTX 2080Ti, served by Ollama.
- Eight tool providers: Home Assistant, timers and reminders, RAG over my own
  equipment manuals (Qdrant + `nomic-embed-text`), Frigate cameras, web search,
  notes, memory, calculator.
- Hard turns escalate to Claude, driving the same tools.
- Voice in and out on the same box: faster-whisper for STT, Kokoro for TTS.
- Two surfaces, a browser PWA and a Raspberry Pi satellite, share one
  server-owned session, so a thread started on one continues on the other.
- Replies stream to the client as Server-Sent Events.

Plain chat, timers, math and memory are always on; the rest switch on when their
env vars are set, and the router hands the model only the handful of tools a
given turn needs. Keeping that list short is what holds accuracy up as the
catalog grows, and it gives the trace a shape worth looking at.

```
backend/app/
  main.py       all /api routes; wires up the plugin registry
  skills.py     per-turn tool routing (which skills a message needs)
  tracing.py    the whole Arize instrumentation surface
  config.py     env-driven settings (every NOVA_VOICE_* knob)
  services/     upstream clients: stt, llm, tts, claude, home_assistant,
                obsidian, searxng, qdrant, embeddings, notify, frigate
  plugins/      tool providers: home_assistant, obsidian, timers, websearch,
                memory, calculator, manuals, cameras (plus base, registry)
backend/evals/  dataset, experiment, five evaluators, traffic generator
frontend/       the PWA (no build step: plain HTML/CSS/JS, ES modules)
```

Arize's 30+ advertised integrations covered 1 of the 9 components that needed
measurement, and a second one *looked* covered and was not. After the manual
instrumentation in [`backend/app/tracing.py`](backend/app/tracing.py) coverage is
7 of 9; STT and TTS remain, both optional. Tracing also turned up two production
bugs that had been live for weeks.

---

## Why this application

Two properties dominate everything that follows:

1. **There is no agent framework.** The tool loop is hand-written in
   [`backend/app/services/llm.py`](backend/app/services/llm.py).
2. **The local model is called over raw HTTP.** `httpx` → Ollama's native
   `POST /api/chat`, chosen deliberately so the app can pass `think: false`. The
   `ollama` Python client is not a dependency.

Both are ordinary production choices. The onboarding anticipates neither one.

| Property of Nova Voice | Why an observability vendor feels it |
|---|---|
| Raw `httpx` to Ollama's native API (no OpenAI SDK, no LangChain) | No auto-instrumentor fires; forces the manual path |
| The Claude escalation leg *does* use the `anthropic` SDK | One leg auto-instruments, one doesn't, so span shapes are inconsistent in a single trace |
| A deterministic keyword router runs before any model call | Not an LLM span, not a tool span; OpenInference has no natural home for it |
| Two models in one trace (qwen3 → Claude, fast/deep tiers) | "Which model answered, and should it have?" has no native view |
| RAG over PDFs via Qdrant + `nomic-embed-text`, raw REST | RETRIEVER spans, citation grounding, chunk relevance |
| Answers stream to the browser as SSE | A root span must stay open across an async generator's yields |
| Two surfaces (browser PWA + a Pi satellite) sharing one server-owned session | Session-scoped tracing across clients |
| LAN-only unless I turn escalation on, so leaving the network is a deliberate act | Shipping traces to a SaaS is a governance decision, not a checkbox |

### The coverage matrix

The nine things needing measurement, against what onboarding offered:

| # | Component | Where | Span kind | Onboarding offers | Reality |
|---|---|---|---|---|---|
| 1 | Turn / session root | `main.py::api_chat` | AGENT | Nothing (assumes a framework emits it) | Hand-rolled |
| 2 | Skill router | `registry.route()` | CHAIN | Nothing | Hand-rolled |
| 3 | Local LLM (qwen3) | `services/llm.py` (httpx) | LLM | Ollama tile, a false positive | Hand-rolled |
| 4 | Tool dispatch | `registry.execute()` | TOOL | Nothing | Hand-rolled |
| 5 | Manuals RAG | `plugins/manuals.py`, `services/qdrant.py` | RETRIEVER | Nothing | Hand-rolled |
| 6 | Escalation decision | `escalation.py` | CHAIN | Nothing | Hand-rolled |
| 7 | Claude escalation | `services/claude.py` | LLM | AnthropicInstrumentor (works) | Paste the snippet |
| 8 | STT (faster-whisper) | `services/stt.py` | n/a | Nothing | Optional, not done |
| 9 | TTS (Kokoro) | `services/tts.py` | n/a | Nothing | Optional, not done |

The Ollama tile instruments the `ollama` Python *client*, not Ollama's HTTP API,
so it misses this app's primary model, and you cannot discover that without
installing the package and observing silence.

---

## The trace shape

```
chat.turn                            AGENT      session.id = conversation_id
│                                               metadata: source(web|satellite),
│                                               mode, auto_escalate, history_turns
├── route                            CHAIN      in: recent user texts
│                                               out: skill_names, tools offered
├── llm.local                        LLM        model, system note, messages,
│   │                                           tool specs, token counts, TTFT
│   ├── tool.timers__start_timer     TOOL       args + result
│   ├── tool.manuals__search_manuals RETRIEVER  query, k, per-doc score + page
│   └── llm.local (round 2)          LLM        the post-tool round
├── escalation.decide                CHAIN      upfront | reactive | none + trigger
└── llm.claude                       LLM        tier(fast|deep), model, tool loop
```

Three design rules shaped [`tracing.py`](backend/app/tracing.py):

- **Inert unless configured.** Nothing is imported at module scope, so the
  hermetic test suite passes with none of the OTel packages installed. A missing
  key degrades every helper to a no-op with the same signature.
- **Explicit span parents, not implicit context.** OTel's `start_as_current_span`
  relies on a contextvar that does not survive an async generator's `yield`
  points, and a whole turn streams out of one. Spans are created with an explicit
  `parent=` and the caller holds the handle. More plumbing, but it survives the
  yields.
- **Never break a turn.** Every tracing failure is logged and swallowed.
  Observability must not be able to take the assistant down.

---

## Read these files, in this order

| File | What it shows |
|---|---|
| [`backend/app/tracing.py`](backend/app/tracing.py) | The entire instrumentation surface: span kinds, OpenInference attributes hand-spelled from the spec, the explicit-parent model |
| [`backend/app/main.py`](backend/app/main.py) (`api_chat`) | The traced turn: root span, router child, `traced_stream()` closing the root even if the client disconnects |
| [`backend/app/services/llm.py`](backend/app/services/llm.py) | LLM spans on the hand-written tool loop: token counts, TTFT, tool and retriever children |
| [`backend/app/services/claude.py`](backend/app/services/claude.py) | The `activate()` bridge that nests auto-instrumented Anthropic spans under a hand-rolled parent, plus the escalation bug below |
| [`backend/app/services/searxng.py`](backend/app/services/searxng.py) | Failure-as-absence, found in my own code by reading a trace |
| [`backend/evals/generate_traffic.py`](backend/evals/generate_traffic.py) | How the traffic under observation was produced |
| [`backend/evals/arize_experiment.py`](backend/evals/arize_experiment.py) | Dataset upload, experiment run, five evaluators, per-tag scorecard |
| [`backend/tests/test_tracing.py`](backend/tests/test_tracing.py) | Tests pinning the attribute spellings, because a typo is silent in the UI |

---

## What tracing found

Two bugs that had been live for weeks. Both turned up in the traces.

### 1. Every reactive escalation had been failing

`claude.agent_reply` set `thinking: {"type": "adaptive"}` unconditionally, but
adaptive thinking is a 4.6-generation feature: `claude-opus-4-8` (the DEEP tier)
accepts it, while `claude-haiku-4-5` (the FAST tier, used on the reactive path)
rejects it with `400 invalid_request_error`.

The asymmetry is why it went unnoticed. The failure only ever appeared on the
path that fires when something else has already gone wrong, so it read as the
local model failing rather than the retry.

What surfaced it was 4 of 28 traced turns coming back ESCALATED with an error
body.

Fix: `_thinking_for(model)` in [`services/claude.py`](backend/app/services/claude.py),
pinned by [`tests/test_claude_thinking.py`](backend/tests/test_claude_thinking.py).

### 2. A dead search backend read as "no results found"

SearXNG returns HTTP 200 with `results: []` both when nothing matched and when
every upstream engine refused the query; only `unresponsive_engines` tells them
apart, and we were discarding it. So every `web_search` had been reporting "no
results found" while the search backend was entirely down. The model took that at
face value and answered from its own knowledge, which puts an ungrounded answer
in the same shape as a grounded one.

I found it while tracing a turn that said "web search is coming up empty right
now."

Fix in [`services/searxng.py`](backend/app/services/searxng.py), covered by
[`tests/test_websearch_degraded.py`](backend/tests/test_websearch_degraded.py).

The second one is the shape several of the platform findings below also take: a
break that renders identically to a legitimate empty state. I wasn't looking for
it in my own code, which is part of why I trust the pattern.

---

## Evaluation

I also ran the corpus as an Arize dataset + experiment, and put one LLM-as-judge
evaluator online against live spans.

- **Dataset** `nova-agent-cases`: 55 hand-written cases, tagged
  `safety` / `honesty` / `grounding` / `followup` / `negative` / `hard`
  ([`backend/evals/cases.jsonl`](backend/evals/cases.jsonl)).
- **Experiment task** drives the real routed agent loop: same router, same
  system note, same `llm.agent_reply` the browser hits. Only tool execution is
  faked; the calculator runs for real, being pure.
- **Five evaluators**, reusing the tolerance rules
  [`backend/evals/README.md`](backend/evals/README.md) already documents.

```
action_honesty    100.0% n=55      personalization   75.0% n=1
skill_routing     100.0% n=55      safety            91.7% n=4
argument_accuracy  97.4% n=38      hard              92.7% n=11
tool_selection     94.4% n=54      grounding         93.6% n=18
answer_grounding   91.7% n=48      negative         100.0% n=5
```

The eval found a safety bug in the assistant. A case stores a peanut
allergy in memory, asks for a snack suggestion, and the reply contains "peanut",
caught by an `answer_excludes` guard. Exactly the failure you want found by a
corpus rather than by someone eating it.

An ablation then tested a system-prompt directive that had been in place for
months on the strength of an anecdote:

| Evaluator | Baseline | `_TOOL_USE_NOTE` stripped | Δ |
|---|---|---|---|
| argument_accuracy | 97.4% | 92.1% | -5.3pp |
| tool_selection | 94.4% | 90.7% | -3.7pp |
| answer_grounding | 91.7% | 89.6% | -2.1pp |

Directionally it earns its place: every non-saturated metric moved down, none up.
It is suggestive rather than conclusive, though. n=55, one run,
`temperature: 0.7`; a 3.7pp shift on n=54 is two cases. Three independent metrics
agreeing is what makes it credible, not any single delta. Nothing in the
experiment surface pushed me toward the repeated runs a firm answer would need.

The per-tag scorecard on the right-hand side above exists only because I wrote a
manual UUID join ([`backend/evals/arize_report.py`](backend/evals/arize_report.py));
experiment results drop every dataset column, including the tags that were the
entire point of building the corpus that way.

---

## Running it

You can't run the whole thing. It needs a GPU box with Ollama, faster-whisper and
Kokoro, plus Home Assistant, Qdrant and Frigate on the LAN. Which is more or less
the point.

What you can run without any of that:

```bash
# The hermetic test suite: 197 tests, stdlib-only except httpx/anthropic
cd backend && python -m unittest discover -s tests

# Prove the exporter and the span shape without Nova's services up
ARIZE_SPACE_ID=xxx ARIZE_API_KEY=yyy python -m evals.arize_smoke

# With an Ollama anywhere on your network, drive the real loop
python -m evals.arize_smoke --live
```

Tracing is off by default and gated on both `ARIZE_SPACE_ID` and
`ARIZE_API_KEY` being set, because traces carry prompts, tool arguments and tool
results off the LAN. See [`.env.example`](.env.example).
