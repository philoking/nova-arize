"""Deciding when a local (qwen3) turn "struggled" and should escalate to Claude.

Auto-escalation runs the local model first and only hands the turn to Claude when
qwen3 fell short - so we pay for the cloud brain only when it earns its keep. The
signal is deterministic and text-based: an *empty* answer, or a reply that **punts**
instead of doing the task (asks the user's permission to do something it has a tool
for, defers to the manufacturer / a website, or says it lacks access). These are
exactly the qwen3 failure modes observed on multi-step tool turns (#63).

Kept conservative on purpose: a false positive costs Claude tokens on a turn that
was actually fine, so the patterns match clear deferrals, not benign hedging like
"let me know if you'd like more." Pure/deterministic so it's unit-tested without a
model.
"""

from __future__ import annotations

import re

# Phrases that signal the model deferred the task rather than completing it.
_PUNT_PATTERNS = [
    r"would you like me to\b",
    r"do you want me to\b",
    r"shall i\b",
    r"should i (search|look|check|go ahead)\b",
    r"i recommend (checking|consulting|contacting|visiting|searching)\b",
    r"i(?:'d| would) recommend (checking|consulting|contacting|visiting)\b",
    r"you (?:can|could|may|might|should|'?ll need to|will need to) (?:check|consult|contact|look|search|visit|refer)\b",
    r"you(?:'?ll| will) (?:want|need) to (?:check|consult|contact|look|search|visit)\b",
    r"consult (?:the )?(?:manufacturer|manual|documentation|website)\b",
    r"manufacturer'?s? (?:website|support|documentation)\b",
    r"contact .{0,25}(?:support|customer service|manufacturer)\b",
    r"check (?:the )?(?:manufacturer|website|manual|documentation)\b",
    r"i don'?t have (?:access|the manual|information|specific|any information|details)\b",
    r"i (?:cannot|can'?t) (?:find|access|provide|retrieve)\b",
    r"i(?:'?m| am) (?:unable|not able) to (?:find|access|provide|retrieve|determine)\b",
    r"i don'?t have enough (?:information|detail)\b",
    r"for (?:precise|exact|specific|accurate) .{0,40}(?:consult|contact|check|visit|refer)\b",
]
_PUNT_RE = re.compile("|".join(_PUNT_PATTERNS), re.IGNORECASE)


def is_struggle(reply: str, tool_calls: list[dict] | None = None) -> bool:
    """True if the local turn should be retried on Claude.

    ``reply`` is the assistant's final text; ``tool_calls`` are the tools it ran
    this turn (reserved for future signals). Escalate on an empty answer or a clear
    punt/deferral.
    """
    text = (reply or "").strip()
    if not text:
        return True
    return bool(_PUNT_RE.search(text))


# Intents worth sending straight to Claude: the local model would answer, but
# poorly, and wouldn't punt (so `is_struggle` never fires). Conservative on purpose,
# since a false positive spends tokens on a turn qwen3 could have handled.
_UPFRONT_PATTERNS = [
    # Explicit ask for Claude.
    r"\b(ask|use|switch to|escalate to) claude\b",
    r"\bask (anthropic|the cloud)\b",
    # Long-form writing - anchored to a written artifact so casual "write" ("write
    # a timer") doesn't trip it.
    r"\b(write|draft|compose|rewrite|proofread|polish|edit)\b[^.?!]{0,40}\b"
    r"(email|essay|letter|message|note|story|poem|plan|report|summary|proposal|"
    r"paragraph|article|blog|post|script|cover letter|resume|cv|speech|outline|memo)\b",
    # Research / deep reasoning.
    r"\bresearch\b",
    r"\bdeep[- ]dive\b",
    r"\bin (depth|detail)\b",
    r"\b(thorough(ly)?|comprehensive|detailed|in-depth)\b[^.?!]{0,30}\b"
    r"(analysis|explanation|breakdown|comparison|overview|guide|answer|write-?up|report)\b",
    r"\bstep[- ]by[- ]step\b",
    r"\bpros and cons\b",
    r"\b(analy[sz]e|brainstorm)\b",
    r"\bcompare\b[^.?!]{0,40}\b(and|versus|vs\.?|to|with)\b",
]
_UPFRONT_RE = re.compile("|".join(_UPFRONT_PATTERNS), re.IGNORECASE)


def should_escalate_upfront(message: str) -> bool:
    """True if the *incoming request* should go straight to Claude (research,
    long-form writing, deep reasoning, or an explicit "ask Claude") - skipping the
    local pass qwen3 would answer poorly without ever signalling a struggle."""
    return bool(_UPFRONT_RE.search(message or ""))
