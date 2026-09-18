"""Tests for the SQLite layer behind persistent libraries."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library_fakes import make_blocks
from rag_core.chunker import chunk_document
from rag_core.library_db import DocumentRecord, LibraryDB


def record(doc_id="d1", **overrides) -> DocumentRecord:
    base = dict(
        doc_id=doc_id, filename=f"{doc_id}.pdf", file_size=100, doc_type=".pdf",
        page_or_slide_count=3, file_sha256=f"fsha-{doc_id}", text_sha256=f"tsha-{doc_id}",
        collection="General", added_at="2026-09-17T10:00:00", chunk_count=0, complete=True,
    )
    base.update(overrides)
    return DocumentRecord(**base)


@pytest.fixture
def db(tmp_path):
    d = LibraryDB(tmp_path / "library.db")
    yield d
    d.close()


class TestSchema:
    def test_wal_mode_and_schema_version(self, db):
        mode = db.connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        version = db.connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        assert version == "1"

    def test_reopen_keeps_data(self, tmp_path):
        first = LibraryDB(tmp_path / "library.db")
        first.insert_document(record())
        first.close()
        second = LibraryDB(tmp_path / "library.db")
        assert second.get_document("d1").filename == "d1.pdf"
        second.close()


class TestDocuments:
    def test_round_trip(self, db):
        db.insert_document(record(title="Foxes", title_source="llm", doi="10.1/ABC"))
        got = db.get_document("d1")
        assert got.title == "Foxes" and got.title_source == "llm"
        assert got.doi == "10.1/ABC" and got.complete is True

    def test_incomplete_hidden_from_lists_and_lookups(self, db):
        db.insert_document(record(complete=False))
        assert db.list_documents() == []
        assert db.find_by_file_hash("fsha-d1") is None
        assert [d.doc_id for d in db.incomplete_documents()] == ["d1"]
        assert [d.doc_id for d in db.list_documents(include_incomplete=True)] == ["d1"]

    def test_mark_complete(self, db):
        db.insert_document(record(complete=False))
        db.mark_complete("d1", chunk_count=7)
        got = db.get_document("d1")
        assert got.complete and got.chunk_count == 7

    def test_hash_lookups(self, db):
        db.insert_document(record())
        assert db.find_by_file_hash("fsha-d1").doc_id == "d1"
        assert db.find_by_text_hash("tsha-d1").doc_id == "d1"
        assert db.find_by_file_hash("nope") is None

    def test_doi_lookup_is_case_insensitive(self, db):
        db.insert_document(record(doi="10.1000/ABC.123"))
        assert db.find_by_doi("10.1000/abc.123").doc_id == "d1"

    def test_title_year_lookup(self, db):
        db.insert_document(record(title="A Study: of Foxes!", year="2021"))
        assert db.find_by_title_year("a study of foxes", "2021").doc_id == "d1"
        assert db.find_by_title_year("a study of foxes", "2020") is None

    def test_update_details(self, db):
        db.insert_document(record())
        db.update_details("d1", title="New Title", title_source="user", year="1999")
        got = db.get_document("d1")
        assert got.title == "New Title" and got.title_source == "user" and got.year == "1999"
        assert db.find_by_title_year("new title", "1999").doc_id == "d1"

    def test_update_details_rejects_unknown_field(self, db):
        db.insert_document(record())
        with pytest.raises(ValueError):
            db.update_details("d1", filename="evil.pdf")

    def test_collections(self, db):
        db.insert_document(record("d1", collection="Week 1"))
        db.insert_document(record("d2", collection="Week 2"))
        db.insert_document(record("d3", collection="Week 1"))
        assert db.collections() == ["Week 1", "Week 2"]
        assert {d.doc_id for d in db.list_documents(collection="Week 1")} == {"d1", "d3"}

        db.set_collection("d2", "Week 1")
        assert db.collections() == ["Week 1"]

        affected = db.rename_collection("Week 1", "Unit A")
        assert sorted(affected) == ["d1", "d2", "d3"]
        assert db.collections() == ["Unit A"]


class TestChunks:
    def test_round_trip_preserves_fields_and_dedupes_parents(self, db):
        db.insert_document(record())
        chunks = chunk_document(make_blocks(3), "d1.pdf", ".pdf", doc_key="d1")
        chunks[0].extra["context_header"] = "About foxes."
        chunks[0].extra["collection"] = "General"
        db.insert_chunks("d1", chunks)

        loaded = db.load_chunks("d1")
        assert [c.chunk_id for c in loaded] == [c.chunk_id for c in chunks]
        for orig, back in zip(chunks, loaded):
            assert back.text == orig.text
            assert back.parent_text == orig.parent_text
            assert back.parent_id == orig.parent_id
            assert back.page == orig.page
            assert back.section == orig.section
            assert back.source_file == "d1.pdf"
            assert back.doc_type == ".pdf"
        assert loaded[0].extra == {"context_header": "About foxes.", "collection": "General"}

        parents = db.connection.execute("SELECT COUNT(*) FROM parents").fetchone()[0]
        assert parents == len({c.parent_id for c in chunks})

    def test_load_all_and_chunk_ids(self, db):
        for doc in ("d1", "d2"):
            db.insert_document(record(doc))
            db.insert_chunks(doc, chunk_document(make_blocks(2, doc), f"{doc}.pdf", ".pdf", doc_key=doc))
        assert len(db.load_chunks()) == 4
        assert len(db.chunk_ids("d2")) == 2

    def test_delete_cascades(self, db):
        db.insert_document(record())
        db.insert_chunks("d1", chunk_document(make_blocks(2), "d1.pdf", ".pdf", doc_key="d1"))
        db.delete_document("d1")
        assert db.get_document("d1") is None
        assert db.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert db.connection.execute("SELECT COUNT(*) FROM parents").fetchone()[0] == 0
