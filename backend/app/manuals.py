"""Shop-manual library - upload PDFs and make them searchable (#63).

The registry of uploaded manuals lives in the shared SQLite DB (``db.py``); the
original PDFs sit on the ``/data`` volume (``manuals_dir``); the *vectors* live in
Qdrant. This module owns the ingest pipeline that ties them together:

    PDF bytes → extract text per page (pypdf) → chunk → embed (Ollama) →
    upsert to Qdrant (payload tags each chunk with its doc + page)

and the inverse (delete a manual: drop its Qdrant points, its file, and its row).
The ``search_manuals`` tool (see ``plugins/manuals.py``) embeds the question and
queries Qdrant; this module is the write side plus the document registry the
Settings ▸ Manuals tab lists.

Text extraction is best-effort: a scanned, image-only PDF yields no text, so the
manual is recorded with status ``empty`` and the upload response says so rather
than silently storing nothing searchable. OCR is out of scope for now.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

from . import db
from .config import settings
from .logstore import log
from .services import embeddings, qdrant

# Embed in batches so a big manual doesn't build one enormous request body.
_EMBED_BATCH = 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS manuals (
    id         TEXT PRIMARY KEY,
    filename   TEXT NOT NULL,               -- original upload name
    title      TEXT NOT NULL,               -- display name (derived, editable)
    bytes      INTEGER NOT NULL DEFAULT 0,
    pages      INTEGER NOT NULL DEFAULT 0,
    chunks     INTEGER NOT NULL DEFAULT 0,
    -- 'pending'  → staged, waiting for the background worker
    -- 'processing' → currently being extracted/embedded/indexed
    -- 'ready' | 'empty' | 'error' → terminal outcomes
    status     TEXT NOT NULL DEFAULT 'ready',
    error      TEXT,
    created_at INTEGER NOT NULL
);
"""

_ready = False

# Background ingest queue: stage() drops a doc_id in, run_worker() indexes them one
# at a time. Single-consumer on purpose - the embedder is one GPU, so a bulk import
# of many manuals should trickle through, not stampede Ollama. Created lazily so the
# module imports without a running loop (tests, RAG-less deploys).
_WORK_QUEUE: "asyncio.Queue[str] | None" = None


def _now() -> int:
    return int(time.time())


def init() -> None:
    global _ready
    with db.lock:
        db.conn().executescript(_SCHEMA)
        db.conn().commit()
        _ready = True


def _db():
    if not _ready:
        init()
    return db.conn()


# PDF text extraction + chunking (pure; unit-tested without a live service)

def extract_pages(data: bytes) -> list[str]:
    """Text of each page, in order. Pages with no extractable text come back ``""``.

    ``pypdf`` is imported lazily so importing this module (and the chunking helper)
    never requires the dependency - handy for tests and for a deploy without RAG.
    """
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "") for page in reader.pages]


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window chunks of ``text`` (whitespace-collapsed), breaking on a
    space near the window edge so words aren't split mid-token. Windows overlap by
    ``overlap`` chars so a fact straddling a boundary still lands whole in one."""
    text = " ".join((text or "").split())
    if not text:
        return []
    size = max(size, 1)
    overlap = max(0, min(overlap, size - 1))
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # prefer a space boundary in the back half of the window
            sp = text.rfind(" ", start + size // 2, end)
            if sp != -1:
                end = sp
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(pages: list[str], size: int, overlap: int) -> list[dict]:
    """Flatten pages into ``[{"page", "text"}, …]`` chunks (page is 1-indexed)."""
    out: list[dict] = []
    for i, page_text in enumerate(pages, start=1):
        for piece in _chunk_text(page_text, size, overlap):
            out.append({"page": i, "text": piece})
    return out


def _title_from_filename(filename: str) -> str:
    base = os.path.basename(filename or "").rsplit(".", 1)[0]
    return " ".join(base.replace("_", " ").replace("-", " ").split()) or "Untitled manual"


# Registry (SQLite)

def _record(doc_id, filename, title, nbytes, pages, chunks, status, error, now) -> None:
    with db.lock:
        _db().execute(
            "INSERT OR REPLACE INTO manuals "
            "(id, filename, title, bytes, pages, chunks, status, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, filename, title, nbytes, pages, chunks, status, error, now),
        )
        _db().commit()


def _set_status(doc_id: str, status: str, error: str | None = None) -> None:
    with db.lock:
        _db().execute(
            "UPDATE manuals SET status = ?, error = ? WHERE id = ?", (status, error, doc_id))
        _db().commit()


def get(doc_id: str) -> dict | None:
    with db.lock:
        row = _db().execute("SELECT * FROM manuals WHERE id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def list_all() -> list[dict]:
    """Every uploaded manual, most-recent first (backs the Settings tab)."""
    with db.lock:
        rows = _db().execute("SELECT * FROM manuals ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def _path(doc_id: str) -> str:
    return os.path.join(settings.manuals_dir, f"{doc_id}.pdf")


def file_path(doc_id: str) -> str:
    """On-disk path of a manual's stored PDF (the original bytes are kept on the
    ``/data`` volume so the app can serve them back - e.g. an in-app PDF viewer)."""
    return _path(doc_id)


# Ingest: stage (fast) + background index
# Upload is decoupled from indexing so a bulk import returns immediately and the
# heavy work (extract → embed → Qdrant) trickles through one worker. stage() writes
# the PDF and records it 'pending'; run_worker() picks it up and runs _index().

def _queue() -> "asyncio.Queue[str]":
    global _WORK_QUEUE
    if _WORK_QUEUE is None:
        _WORK_QUEUE = asyncio.Queue()
    return _WORK_QUEUE


async def stage(filename: str, data: bytes, title: str | None = None) -> dict:
    """Persist a PDF and enqueue it for background indexing. Returns the ``pending``
    registry row immediately - the actual extract/embed/index happens in
    ``run_worker``. The bytes are kept on disk so indexing survives a restart.
    """
    doc_id = uuid.uuid4().hex
    title = (title or "").strip() or _title_from_filename(filename)
    now = _now()

    os.makedirs(settings.manuals_dir, exist_ok=True)
    with open(_path(doc_id), "wb") as f:
        f.write(data)

    _record(doc_id, filename, title, len(data), 0, 0, "pending", None, now)
    _queue().put_nowait(doc_id)
    return get(doc_id)


async def _index(doc_id: str) -> dict | None:
    """Extract, chunk, embed, and index a staged manual, driving its status to a
    terminal state (``ready``/``empty``/``error``). Reads the PDF back from disk so
    a doc requeued after a restart still indexes. Never raises: failures land on the
    record's ``status``/``error`` so the library still shows the row.
    """
    rec = get(doc_id)
    if rec is None:
        return None
    filename, title, created_at = rec["filename"], rec["title"], rec["created_at"]

    try:
        with open(_path(doc_id), "rb") as f:
            data = f.read()
    except OSError as exc:
        log.warning("manual %s: stored file missing: %s", doc_id, exc)
        _record(doc_id, filename, title, rec["bytes"], 0, 0, "error", f"stored file missing: {exc}", created_at)
        return get(doc_id)

    nbytes = len(data)
    _set_status(doc_id, "processing")

    try:
        pages = await asyncio.to_thread(extract_pages, data)
    except Exception as exc:  # noqa: BLE001 - pypdf raises a variety of errors
        log.warning("manual %r: could not read PDF: %s", filename, exc)
        _record(doc_id, filename, title, nbytes, 0, 0, "error", f"could not read PDF: {exc}", created_at)
        return get(doc_id)

    chunks = chunk_pages(pages, settings.manuals_chunk_chars, settings.manuals_chunk_overlap)
    if not chunks:
        _record(doc_id, filename, title, nbytes, len(pages), 0, "empty",
                "no extractable text — a scanned/image-only PDF needs OCR (not supported)", created_at)
        return get(doc_id)

    try:
        points: list[dict] = []
        for i in range(0, len(chunks), _EMBED_BATCH):
            batch = chunks[i:i + _EMBED_BATCH]
            vectors = await embeddings.embed([c["text"] for c in batch])
            if i == 0:
                await qdrant.ensure_collection(len(vectors[0]))
            for c, vec in zip(batch, vectors):
                points.append({
                    "id": str(uuid.uuid4()),
                    "vector": vec,
                    "payload": {
                        "doc_id": doc_id, "doc_title": title, "filename": filename,
                        "page": c["page"], "text": c["text"],
                    },
                })
        await qdrant.upsert(points)
    except Exception as exc:  # noqa: BLE001 - embed/Qdrant network + data errors
        log.error("manual %r: indexing failed: %s", filename, exc)
        _record(doc_id, filename, title, nbytes, len(pages), 0, "error", f"indexing failed: {exc}", created_at)
        return get(doc_id)

    _record(doc_id, filename, title, nbytes, len(pages), len(chunks), "ready", None, created_at)
    log.info("manual indexed: %r — %d pages, %d chunks", title, len(pages), len(chunks))
    return get(doc_id)


def requeue_pending() -> int:
    """Re-enqueue any manuals left ``pending``/``processing`` by a crash or restart
    (their PDF is still on disk). Called once at startup, before the worker runs."""
    with db.lock:
        rows = _db().execute(
            "SELECT id FROM manuals WHERE status IN ('pending', 'processing')").fetchall()
    for r in rows:
        _queue().put_nowait(r["id"])
    if rows:
        log.info("manuals: requeued %d unfinished import(s) for indexing", len(rows))
    return len(rows)


async def run_worker() -> None:
    """Consume the ingest queue forever, indexing one manual at a time. Started as a
    background task at app startup. One bad doc never kills the loop."""
    q = _queue()
    while True:
        doc_id = await q.get()
        try:
            await _index(doc_id)
        except Exception as exc:  # noqa: BLE001 - belt-and-suspenders; _index already guards
            log.error("manual %s: unexpected worker error: %s", doc_id, exc)
            _set_status(doc_id, "error", f"indexing failed: {exc}")
        finally:
            q.task_done()


async def delete(doc_id: str) -> bool:
    """Remove a manual everywhere: its Qdrant points, its file, and its row."""
    if get(doc_id) is None:
        return False
    try:
        await qdrant.delete_doc(doc_id)
    except Exception as exc:  # noqa: BLE001 - best-effort; still drop the local record
        log.warning("manual %s: Qdrant delete failed (dropping record anyway): %s", doc_id, exc)
    try:
        os.remove(_path(doc_id))
    except OSError:
        pass
    with db.lock:
        _db().execute("DELETE FROM manuals WHERE id = ?", (doc_id,))
        _db().commit()
    return True
