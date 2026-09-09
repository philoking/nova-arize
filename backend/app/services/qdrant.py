"""Qdrant vector-store client for the shop-manual RAG (#63).

Talks to an existing Qdrant instance over its REST API with ``httpx`` - the same
thin-client style as every other service here (no heavyweight ``qdrant-client``
dependency). One collection (``NOVA_QDRANT_COLLECTION``, default ``nova_manuals``)
holds every manual's chunks; each point carries a ``doc_id`` payload so a whole
manual can be deleted in one filtered call.

The collection is created lazily on the first upsert, sized to the embedding
model's actual vector length (so switching embed models just works). Cosine
distance matches nomic-embed-text. Every call returns data or raises
``httpx.HTTPError`` - callers turn that into a user-facing message.
"""

from __future__ import annotations

import httpx

from ..config import settings


def _client() -> httpx.AsyncClient:
    headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
    return httpx.AsyncClient(
        base_url=settings.qdrant_url,
        headers=headers,
        timeout=settings.request_timeout,
        verify=settings.qdrant_verify_ssl,
    )


def _collection() -> str:
    return settings.qdrant_collection


async def collection_exists() -> bool:
    async with _client() as c:
        r = await c.get(f"/collections/{_collection()}")
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True


async def ensure_collection(dim: int) -> None:
    """Create the collection (cosine, ``dim``-sized) if it doesn't already exist."""
    if await collection_exists():
        return
    async with _client() as c:
        r = await c.put(
            f"/collections/{_collection()}",
            json={"vectors": {"size": dim, "distance": "Cosine"}},
        )
        r.raise_for_status()


async def upsert(points: list[dict]) -> None:
    """Upsert points: ``[{"id", "vector", "payload"}, …]``. No-op on empty."""
    if not points:
        return
    async with _client() as c:
        r = await c.put(
            f"/collections/{_collection()}/points",
            params={"wait": "true"},
            json={"points": points},
        )
        r.raise_for_status()


async def search(vector: list[float], *, top_k: int, min_score: float = 0.0) -> list[dict]:
    """Nearest chunks to ``vector``. Returns ``[{"score", "payload"}, …]`` above
    ``min_score``. An empty/absent collection yields ``[]`` rather than raising."""
    async with _client() as c:
        r = await c.post(
            f"/collections/{_collection()}/points/search",
            json={"vector": vector, "limit": top_k, "with_payload": True},
        )
        if r.status_code == 404:
            return []  # collection not created yet (nothing uploaded)
        r.raise_for_status()
        hits = r.json().get("result") or []
    return [
        {"score": h.get("score", 0.0), "payload": h.get("payload") or {}}
        for h in hits
        if h.get("score", 0.0) >= min_score
    ]


async def delete_doc(doc_id: str) -> None:
    """Delete every point belonging to one manual (by its ``doc_id`` payload)."""
    async with _client() as c:
        r = await c.post(
            f"/collections/{_collection()}/points/delete",
            params={"wait": "true"},
            json={"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}},
        )
        if r.status_code == 404:
            return  # nothing to delete
        r.raise_for_status()


async def health() -> bool:
    """Whether Qdrant is reachable (readyz)."""
    if not settings.qdrant_url:
        return False
    try:
        async with _client() as c:
            # Short probe timeout: the shared client defaults to request_timeout
            # (120s) for real queries, but /api/health fans this out and is polled
            # every 20s - a slow Qdrant must not hang the health check for minutes.
            r = await c.get("/readyz", timeout=5)
            return r.status_code == 200
    except httpx.HTTPError:
        return False
