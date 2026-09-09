"""Unit tests for the shop-manual RAG (#63).

Covers the pure text-chunking (whitespace collapse, overlap, page tagging), the
title derivation, the provider's search contract (with the embedding + Qdrant
calls stubbed), and that the router exposes a ``manuals`` skill. Stdlib only - no
live Ollama/Qdrant and no PDF fixture (extraction is a thin pypdf wrapper).

Run from ``backend/``:  python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

os.environ["NOVA_VOICE_HISTORY_DB"] = os.path.join(tempfile.mkdtemp(), "test_manuals.db")
# A Qdrant URL flips the capability on so the provider reports enabled.
os.environ["NOVA_QDRANT_URL"] = "http://qdrant.test:6333"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config, manuals, skills  # noqa: E402
from app.plugins.manuals import ManualsProvider  # noqa: E402
from app.services import embeddings, qdrant  # noqa: E402


class Chunking(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(manuals._chunk_text("a   b\n\n c", 100, 10), ["a b c"])

    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(manuals._chunk_text("   \n  ", 100, 10), [])

    def test_splits_long_text_into_bounded_overlapping_chunks(self):
        text = " ".join(f"w{i}" for i in range(300))
        chunks = manuals._chunk_text(text, 60, 15)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 60)
        # Nothing lost at the ends.
        self.assertTrue(chunks[0].startswith("w0"))
        self.assertTrue(chunks[-1].endswith("w299"))

    def test_overlap_repeats_boundary_content(self):
        text = " ".join(f"w{i}" for i in range(40))
        chunks = manuals._chunk_text(text, 40, 20)
        # With real overlap, consecutive chunks share at least one token.
        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        self.assertTrue(first_words & second_words)

    def test_chunk_pages_tags_1indexed_pages_and_drops_blanks(self):
        pages = ["first page text", "   ", "third page text"]
        chunks = manuals.chunk_pages(pages, 100, 10)
        self.assertEqual([c["page"] for c in chunks], [1, 3])

    def test_title_from_filename(self):
        self.assertEqual(manuals._title_from_filename("DeWalt_Table-Saw.pdf"), "DeWalt Table Saw")
        self.assertEqual(manuals._title_from_filename("path/to/Ryobi Drill.PDF"), "Ryobi Drill")
        self.assertEqual(manuals._title_from_filename(""), "Untitled manual")


class Provider(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._embed, self._search = embeddings.embed_one, qdrant.search
        self.addCleanup(lambda: setattr(embeddings, "embed_one", self._embed))
        self.addCleanup(lambda: setattr(qdrant, "search", self._search))

    def test_enabled_when_qdrant_url_set(self):
        # config.settings is a frozen, read-once singleton, so its qdrant_url is
        # fixed at whichever import first touched app.config. Under a full test run
        # another module imports config before this one sets NOVA_QDRANT_URL (top of
        # file), freezing it empty. Patch the provider's settings with a fresh
        # instance that re-reads the (now-set) env, so the check is order-independent.
        with mock.patch("app.plugins.manuals.settings", config.Settings()):
            self.assertTrue(ManualsProvider().enabled)

    async def test_search_returns_cited_passages(self):
        async def fake_embed_one(q, is_query=False):
            return [0.1, 0.2, 0.3]

        async def fake_search(vec, top_k, min_score=0.0):
            return [{"score": 0.91, "payload": {
                "doc_id": "abc123", "doc_title": "Table Saw", "page": 12,
                "text": "Loosen the arbor nut."}}]

        embeddings.embed_one = fake_embed_one
        qdrant.search = fake_search
        out = await ManualsProvider().execute("search_manuals", {"query": "change the blade"})
        self.assertEqual(out["results"][0]["manual"], "Table Saw")
        self.assertEqual(out["results"][0]["page"], 12)
        self.assertIn("arbor nut", out["results"][0]["text"])
        # id carries the doc so the client can open the exact PDF at this page.
        self.assertEqual(out["results"][0]["id"], "abc123")

    async def test_search_no_hits_reports_note_not_error(self):
        async def fake_embed_one(q, is_query=False):
            return [0.0, 0.0, 0.0]

        async def fake_search(vec, top_k, min_score=0.0):
            return []

        embeddings.embed_one = fake_embed_one
        qdrant.search = fake_search
        out = await ManualsProvider().execute("search_manuals", {"query": "nonexistent"})
        self.assertEqual(out["results"], [])
        self.assertIn("note", out)
        self.assertNotIn("error", out)

    async def test_empty_query_is_an_error(self):
        out = await ManualsProvider().execute("search_manuals", {"query": "   "})
        self.assertIn("error", out)

    async def test_unknown_tool_is_an_error(self):
        out = await ManualsProvider().execute("bogus", {})
        self.assertIn("error", out)


class Ingest(unittest.IsolatedAsyncioTestCase):
    """The stage → background-index pipeline (#63 bulk import). Extraction and the
    embed/Qdrant calls are stubbed; disk + the SQLite registry are real (temp dirs)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["NOVA_MANUALS_DIR"] = self.tmp
        p = mock.patch("app.manuals.settings", config.Settings())
        p.start(); self.addCleanup(p.stop)
        # Fresh ingest queue per test; start from an empty registry.
        manuals._WORK_QUEUE = None
        self.addCleanup(lambda: setattr(manuals, "_WORK_QUEUE", None))
        manuals.init()
        with manuals.db.lock:
            manuals._db().execute("DELETE FROM manuals")
            manuals._db().commit()
        # Stub the heavy write-path services.
        self._embed, self._ensure, self._upsert = (
            embeddings.embed, qdrant.ensure_collection, qdrant.upsert)
        self.addCleanup(lambda: setattr(embeddings, "embed", self._embed))
        self.addCleanup(lambda: setattr(qdrant, "ensure_collection", self._ensure))
        self.addCleanup(lambda: setattr(qdrant, "upsert", self._upsert))

    async def test_stage_records_pending_persists_file_and_enqueues(self):
        rec = await manuals.stage("Ryobi Drill.pdf", b"%PDF-1.4 fake")
        self.assertEqual(rec["status"], "pending")
        self.assertEqual(rec["title"], "Ryobi Drill")
        self.assertEqual(manuals._queue().qsize(), 1)
        self.assertTrue(os.path.exists(manuals.file_path(rec["id"])))

    async def test_index_drives_pending_to_ready_and_upserts(self):
        upserted = []

        async def fake_embed(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        async def fake_ensure(dim):
            return None

        async def fake_upsert(points):
            upserted.extend(points)

        embeddings.embed, qdrant.ensure_collection, qdrant.upsert = (
            fake_embed, fake_ensure, fake_upsert)
        with mock.patch.object(manuals, "extract_pages", return_value=["torque 45 nm", "page two"]):
            rec = await manuals.stage("Grizzly Lathe.pdf", b"x")
            out = await manuals._index(rec["id"])
        self.assertEqual(out["status"], "ready")
        self.assertEqual(out["pages"], 2)
        self.assertGreater(out["chunks"], 0)
        self.assertTrue(upserted)
        self.assertEqual(upserted[0]["payload"]["doc_title"], "Grizzly Lathe")

    async def test_index_no_text_marks_empty(self):
        with mock.patch.object(manuals, "extract_pages", return_value=["", "   "]):
            rec = await manuals.stage("scanned.pdf", b"x")
            out = await manuals._index(rec["id"])
        self.assertEqual(out["status"], "empty")
        self.assertEqual(out["chunks"], 0)

    async def test_index_missing_stored_file_marks_error(self):
        rec = await manuals.stage("gone.pdf", b"x")
        os.remove(manuals.file_path(rec["id"]))
        out = await manuals._index(rec["id"])
        self.assertEqual(out["status"], "error")

    async def test_requeue_pending_reenqueues_unfinished(self):
        await manuals.stage("a.pdf", b"x")
        await manuals.stage("b.pdf", b"y")
        # Simulate a restart: the in-memory queue is gone but the rows are still pending.
        manuals._WORK_QUEUE = None
        self.assertEqual(manuals.requeue_pending(), 2)
        self.assertEqual(manuals._queue().qsize(), 2)


class Routing(unittest.TestCase):
    def test_manuals_skill_exists_with_the_search_tool(self):
        skill = next((s for s in skills.SKILLS if s.name == "manuals"), None)
        self.assertIsNotNone(skill)
        self.assertIn("manuals__search_manuals", skill.tools)

    def test_how_to_question_routes_to_manuals(self):
        skill = next(s for s in skills.SKILLS if s.name == "manuals")
        hits = skills.deterministic_match(
            "how do I change the blade on my table saw", [skill])
        self.assertIn("manuals", hits)


if __name__ == "__main__":
    unittest.main()
