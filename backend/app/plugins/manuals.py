"""Shop-manual RAG as a tool provider (#63).

Gives the model one tool - ``search_manuals`` - that embeds the user's question
and pulls the closest passages from their uploaded manuals (vectors in Qdrant, via
``services/qdrant.py``; query embeddings via ``services/embeddings.py``). Presented
under the ``manuals`` namespace, so the tool reads as ``manuals__search_manuals``.
Enabled only when a Qdrant URL is configured (``manuals_enabled``).

The write side - uploading, extracting, chunking, indexing - lives in the
top-level ``manuals`` module; this is just the read path the agent loop calls.
"""

from __future__ import annotations

import httpx

from .. import manuals
from ..config import settings
from ..services import embeddings, qdrant

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_manuals",
        "description": (
            "Search the user's own uploaded product manuals and documentation for their tools, "
            "appliances, and equipment. Use this for ANY how-to, setting, spec, part number, "
            "torque/measurement, error code, or maintenance-procedure question about a specific "
            "device the user owns — the answer is in their manual, not your general knowledge. "
            "Include the device's brand/model in your query. Returns passages tagged with the manual "
            "title and page; answer from them and cite which manual and page."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description":
                      "What to look up, including the device — e.g. 'grizzly table saw blade change'."},
        }, "required": ["query"]}}},
]

# Generic words that appear in questions but don't identify a specific piece of
# equipment. Vector search rates all "saw/blade/tool" text alike (0.6-0.7 cosine
# across different machines), so passages are gated on the query and the manual
# title sharing a brand, a device-type noun, or a model number.
_GENERIC_TOKENS = frozenset({
    "saw", "blade", "blades", "tool", "tools", "machine", "size", "sizes", "spec",
    "specs", "specification", "specifications", "part", "parts", "number", "manual",
    "manuals", "model", "use", "uses", "using", "the", "and", "for", "you", "your",
    "what", "which", "how", "does", "do", "with", "from", "kind", "type", "need",
    "want", "change", "replace", "install", "adjust", "set", "get", "this", "that",
    "have", "does", "should", "recommended",
})


def _identifying_tokens(text: str) -> set[str]:
    """Brand / device-type / model-number tokens in ``text`` - the words that name a
    specific device. Anything with a digit (model numbers like g0623x, 10-3061) or a
    non-generic word of length ≥ 3 counts; generic question words do not."""
    norm = "".join(c if c.isalnum() or c.isspace() else " " for c in (text or "").lower())
    out: set[str] = set()
    for w in norm.split():
        if any(ch.isdigit() for ch in w) or (len(w) >= 3 and w not in _GENERIC_TOKENS):
            out.add(w)
    return out


class ManualsProvider:
    name = "manuals"

    @property
    def enabled(self) -> bool:
        return settings.manuals_enabled

    def tool_specs(self) -> list[dict]:
        return TOOL_SPECS

    async def execute(self, tool: str, args: dict) -> dict:
        if tool != "search_manuals":
            return {"error": f"unknown tool {tool!r}"}
        query = (args.get("query") or "").strip()
        if not query:
            return {"error": "query is required"}
        try:
            vector = await embeddings.embed_one(query, is_query=True)
            hits = await qdrant.search(
                vector, top_k=settings.manuals_top_k, min_score=settings.manuals_min_score,
            )
        except httpx.HTTPError as exc:
            return {"error": f"manual search unavailable: {exc}"}

        # Device-token gate: cosine can't tell a band saw from a table saw, so keep
        # a passage only if its manual title shares an identifying token (brand,
        # type or model) with the query. A query that names a device but matches no
        # manual returns empty, driving the ask-for-model / web-search fallback
        # rather than an answer from the wrong manual. Generic queries aren't gated.
        q_ident = _identifying_tokens(query)
        if q_ident:
            hits = [h for h in hits
                    if q_ident & _identifying_tokens(h["payload"].get("doc_title") or "")]

        if not hits:
            if settings.websearch_enabled:
                note = ("No manual on file matches that device. If you know its make and model, call "
                        "web_search NOW and answer from the web (say it's from the web, not their "
                        "manual). If you don't have the model, ask the user for the make and model. "
                        "Do not ask the user for permission to search and do not tell them to check a "
                        "website — do it yourself.")
            else:
                note = "No manual on file matches that device; tell the user you don't have its manual."
            return {"query": query, "results": [], "note": note}
        results = [
            {
                "id": h["payload"].get("doc_id"),  # lets the UI open the exact PDF at this page
                "manual": h["payload"].get("doc_title"),
                "page": h["payload"].get("page"),
                "text": h["payload"].get("text"),
                "score": round(h["score"], 3),
            }
            for h in hits
        ]
        return {"query": query, "results": results}

    def system_note(self) -> str:
        ready = [m for m in manuals.list_all() if m["status"] == "ready"]
        note = (
            " You have search_manuals for the user's own uploaded product manuals. When they ask about "
            "a specific tool or appliance, call search_manuals FIRST. Each passage it returns is tagged "
            "with the manual it came from — use a passage ONLY if that manual is for the exact device "
            "the user asked about. NEVER apply one device's manual to a different device (e.g. don't use "
            "a table-saw manual to answer about a band saw) just because the words are similar. Cite the "
            "manual and page for anything you take from it."
        )
        # Manual-first, then ask-for-model, then web-fallback (with attribution).
        if settings.websearch_enabled:
            note += (
                " If none of the returned passages are from a manual for that exact device, you do NOT "
                "have its manual, and you must never answer from a different device's manual. When "
                "answering accurately needs the specific model (blade sizes, parts, torque, specs) and "
                "the user has not already told you the make and model, your ONLY response is to ask them "
                "for it — a direct question such as 'What's the make and model?' — and stop there: do "
                "NOT call a tool and do NOT say you will search yet. Once the user has given the make and "
                "model (or if the model isn't needed at all), call web_search and answer from the "
                "results, saying plainly that this is from the web, not their manual. Never ask the user "
                "for permission to search and never tell them to look it up themselves or visit a "
                "website — just call web_search. Never call search_manuals again for a device you've "
                "already found has no manual on file."
            )
        else:
            note += (
                " If none of the returned passages are from a manual for that exact device, say you don't "
                "have a manual for it instead of answering from a different one."
            )
        if ready:
            titles = "; ".join(m["title"] for m in ready[:30])
            note += f" Manuals on file: {titles}."
        else:
            note += " No manuals have been uploaded yet, so it will return nothing."
        return note

    async def health(self) -> bool:
        return await qdrant.health()
