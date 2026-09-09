"""Text embeddings via the existing Ollama service.

The shop-manual RAG (#63) needs vectors for both the uploaded manual chunks and
the user's question. Rather than stand up a separate embedding service, we reuse
the Ollama instance already running at ``llm_url`` with a dedicated embed model
(``NOVA_EMBED_MODEL``, default ``nomic-embed-text``) - one ``ollama pull`` on Nova
and nothing new to deploy.

``nomic-embed-text`` is a task-prefixed model: documents should be embedded as
``search_document: <text>`` and queries as ``search_query: <text>``. Matching the
prefixes is what makes retrieval accurate, so ``embed`` applies them for you (skip
via ``prefix=False`` for a model that doesn't use them).

Uses Ollama's ``/api/embed`` batch endpoint. Returns plain lists of floats; raises
``httpx.HTTPError`` on an unreachable/erroring service so callers can surface it.
"""

from __future__ import annotations

import httpx

from ..config import settings

_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=settings.llm_url, timeout=settings.request_timeout)


async def embed(texts: list[str] | str, *, is_query: bool = False, prefix: bool = True) -> list[list[float]]:
    """Embed one or more texts. Returns a list of vectors (one per input).

    ``is_query`` selects the query prefix (else the document prefix) for
    prefix-style models like nomic-embed-text; ``prefix=False`` disables it.
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    if prefix:
        p = _QUERY_PREFIX if is_query else _DOC_PREFIX
        inputs = [p + (t or "") for t in texts]
    else:
        inputs = [t or "" for t in texts]

    async with _client() as c:
        r = await c.post("/api/embed", json={"model": settings.embed_model, "input": inputs})
        r.raise_for_status()
        data = r.json()
    vectors = data.get("embeddings") or []
    if len(vectors) != len(inputs):
        raise httpx.HTTPError(
            f"embedding count mismatch: asked {len(inputs)}, got {len(vectors)} "
            f"(is '{settings.embed_model}' pulled on Ollama?)"
        )
    return vectors


async def embed_one(text: str, *, is_query: bool = False) -> list[float]:
    """Embed a single text and return its vector."""
    vecs = await embed([text], is_query=is_query)
    return vecs[0]


async def health() -> bool:
    """Whether the embed model responds - a tiny probe embedding."""
    if not settings.manuals_enabled:
        return False
    try:
        vec = await embed_one("ok", is_query=True)
        return bool(vec)
    except httpx.HTTPError:
        return False
