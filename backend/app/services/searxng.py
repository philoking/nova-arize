"""SearXNG web-search client + tool definition for the agent loop.

Gives the local model a way to look things up on the public web through a
self-hosted SearXNG instance (so the query stays on-prem - SearXNG is the only
thing that talks to the outside). We use SearXNG's JSON API
(``GET /search?q=…&format=json``), which must be enabled in its ``settings.yml``
(``search.formats: [html, json]``); until it is, that endpoint returns 403.

Exposed tool (bare name; the registry namespaces it under ``web__``):

- ``web_search`` - search the public web, returns titles/snippets/links.

Every call returns a plain dict (errors as data) so the model can react rather
than the loop crashing.
"""

from __future__ import annotations

import httpx

from ..config import settings

# A browser-ish UA + JSON Accept keeps SearXNG's bot limiter from refusing the
# server-side request once the json format is enabled.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; nova-voice/1.0; +https://assist.example.internal)",
    "Accept": "application/json",
}

_SNIPPET_MAX = 300  # keep each result compact; the model only needs the gist

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": (
            "Search the public web for current or external facts — news, weather, "
            "sports, prices, people, recent events, or anything you don't reliably "
            "know. Returns titles, snippets, and links; answer from them in your "
            "own words."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to search for."},
        }, "required": ["query"]}}},
]


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.searxng_url,
        headers=_HEADERS,
        timeout=20,
        verify=settings.searxng_verify_ssl,
    )


async def search(query: str, count: int | None = None) -> dict:
    """Query SearXNG and return a compact, model-friendly result set."""
    count = count or settings.searxng_results
    params = {"q": query, "format": "json", "safesearch": "0"}
    try:
        async with _client() as c:
            r = await c.get("/search", params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            return {"error": "SearXNG refused the JSON request — enable the 'json' "
                             "format in its settings.yml (search.formats)."}
        return {"error": f"web search failed: {exc}"}
    except httpx.HTTPError as exc:
        return {"error": f"web search unreachable: {exc}"}
    except ValueError:
        return {"error": "SearXNG did not return JSON — enable the 'json' format "
                         "in its settings.yml (search.formats)."}

    results = [
        {
            "title": (item.get("title") or "").strip(),
            "url": item.get("url") or "",
            "snippet": (item.get("content") or "").strip()[:_SNIPPET_MAX],
        }
        for item in (data.get("results") or [])[:count]
    ]
    out: dict = {"query": query, "results": results}
    # SearXNG sometimes has an instant answer (Wikidata, calculators, etc.).
    if data.get("answers"):
        out["answers"] = data["answers"]

    if not results and not out.get("answers"):
        # An empty result set has two causes, and SearXNG distinguishes them in
        # unresponsive_engines: nothing matched, or every upstream engine refused
        # (CAPTCHA / rate-limit, routine on a residential IP). Reporting both as
        # "no results" let the model answer from memory and present it as grounded.
        down = [e[0] for e in (data.get("unresponsive_engines") or []) if e]
        if down:
            out["error"] = (
                "web search is unavailable — every upstream engine refused the "
                f"query ({', '.join(sorted(down))}). This is a search backend "
                "failure, NOT an absence of results. Say you could not search; "
                "do not answer as though the web had nothing on the subject."
            )
            out["unresponsive_engines"] = sorted(down)
        else:
            out["note"] = "no results found"
    return out


async def execute_tool(tool: str, args: dict) -> dict:
    if tool == "web_search":
        query = (args.get("query") or "").strip()
        if not query:
            return {"error": "query is required"}
        return await search(query)
    return {"error": f"unknown tool {tool!r}"}


async def health() -> bool:
    """Reachability of the SearXNG instance (homepage 200)."""
    if not settings.searxng_url:
        return False
    try:
        async with _client() as c:
            # Short probe timeout so a slow/unreachable SearXNG can't stall
            # /api/health (the shared client defaults to 20s; polled every 20s).
            r = await c.get("/", timeout=5)
            return r.status_code == 200
    except httpx.HTTPError:
        return False
