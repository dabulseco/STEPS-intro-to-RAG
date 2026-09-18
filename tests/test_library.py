"""Tests for library lifecycle, duplicate checks, recovery and re-embedding."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library_fakes import StubEmbedder, details_llm, make_blocks, processed, silent_llm
from rag_core import library as lib
from rag_core.library_db import DocumentRecord
from rag_core.library_lock import LibraryLock, LibraryLocked
from rag_core.persistent_store import release_chroma


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv(lib.LIBRARY_ROOT_ENV, str(tmp_path / "libs"))
    yield tmp_path / "libs"
    release_chroma()


@pytest.fixture
def library(root):
    path = lib.create_library("BIOL 201", "stub-embed")
    opened = lib.Library.open(path, embedder=StubEmbedder())
    yield opened
    opened.close()


def item(name="smith2021.pdf", data=None, prefix=None):
    prefix = prefix if prefix is not None else name.split(".")[0]
    data = data if data is not None else f"bytes-of-{name}".encode()
    return name, data, processed(name, make_blocks(3, prefix))


def add(library, name="smith2021.pdf", collection="General", ask_llm=None, **kw):
    result = library.check(*item(name, **kw))
    assert result.status == "new", result.message
    return library.add(result, collection, ask_llm=ask_llm)


# -----------------------------------------------------------------------------
class TestLifecycle:
    def test_root_from_env(self, root):
        assert lib.library_root() == root

    def test_create_list_and_read(self, root):
        path = lib.create_library("BIOL 201", "nomic-embed-text", contextual_headers=True)
        assert path == root / "biol-201"
        info = lib.read_info(path)
        assert info.schema_version == 1
        assert info.name == "BIOL 201"
        assert info.embedding_model == "nomic-embed-text"
        assert info.contextual_headers is True
        assert info.chatbot.name == "BIOL 201"
        assert info.chatbot.rerank is True
        assert (path / "library.db").exists()
        assert [i.name for _, i in lib.list_libraries()] == ["BIOL 201"]

    def test_create_records_chat_model(self, root):
        path = lib.create_library("With Model", "m", chat_model="gemma4:e4b")
        assert lib.read_info(path).chatbot.chat_model == "gemma4:e4b"
        assert lib.read_info(lib.create_library("No Model", "m")).chatbot.chat_model == ""

    def test_slug_collisions_get_suffix(self, root):
        first = lib.create_library("Cell Bio", "m")
        second = lib.create_library("cell bio!", "m")
        assert first.name == "cell-bio" and second.name == "cell-bio-2"

    def test_info_round_trip_ignores_unknown_keys(self, root):
        path = lib.create_library("X Lib", "m")
        raw = json.loads((path / "library.json").read_text())
        raw["chatbot"]["future_setting"] = 1
        raw["future_top_level"] = True
        (path / "library.json").write_text(json.dumps(raw))
        info = lib.read_info(path)
        info.chatbot.temperature = 0.1
        lib.write_info(path, info)
        assert lib.read_info(path).chatbot.temperature == 0.1

    def test_rename_changes_display_name_only(self, library):
        library.rename("Cell Biology")
        assert lib.read_info(library.path).name == "Cell Biology"
        assert library.path.name == "biol-201"

    def test_delete_library(self, root):
        path = lib.create_library("Temp", "m")
        opened = lib.Library.open(path, embedder=StubEmbedder())
        add(opened)
        opened.close()
        lib.delete_library(path)
        assert not path.exists()
        assert lib.list_libraries() == []

    def test_retrieval_settings_mapping(self):
        s = lib.ChatbotSettings(name="x", dense_k=30, final_k=7, multi_query=True, rerank=False)
        r = s.retrieval_settings()
        assert (r.dense_k, r.sparse_k, r.final_k, r.multi_query, r.rerank) == (30, 30, 7, True, False)


# -----------------------------------------------------------------------------
class TestLock:
    def test_second_live_process_is_refused(self, root):
        path = lib.create_library("Locked", "m")
        (path / ".lock").write_text(str(os.getppid()))   # a live process that is not us
        with pytest.raises(LibraryLocked):
            lib.Library.open(path, embedder=StubEmbedder())
        with pytest.raises(LibraryLocked):
            lib.delete_library(path)

    def test_stale_lock_is_replaced(self, root):
        path = lib.create_library("Stale", "m")
        (path / ".lock").write_text("999999")            # no such process
        opened = lib.Library.open(path, embedder=StubEmbedder())
        assert (path / ".lock").read_text() == str(os.getpid())
        opened.close()
        assert not (path / ".lock").exists()

    def test_same_process_may_reopen(self, tmp_path):
        lock = LibraryLock(tmp_path / ".lock")
        lock.acquire()
        LibraryLock(tmp_path / ".lock").acquire()
        lock.release()


# -----------------------------------------------------------------------------
class TestCheck:
    def test_new(self, library):
        r = library.check(*item())
        assert (r.status, r.action) == ("new", "add")

    def test_exact_duplicate(self, library):
        add(library)
        r = library.check(*item())
        assert (r.status, r.action) == ("duplicate", "skip")
        assert "smith2021.pdf" in r.message

    def test_same_text_different_bytes(self, library):
        add(library)
        r = library.check(*item("renamed.pdf", data=b"other bytes", prefix="smith2021"))
        assert r.status == "duplicate"

    def test_same_doi_is_possible_duplicate(self, library):
        add(library, ask_llm=details_llm(doi="10.1000/fox.1"))
        name, data, proc = item("preprint.pdf")
        proc["blocks"][0]["text"] += " doi:10.1000/FOX.1"
        proc["content"] = "\n\n".join(b["text"] for b in proc["blocks"])
        r = library.check(name, data, proc)
        assert (r.status, r.action) == ("possible_duplicate", "skip")
        assert r.existing is not None and "DOI" in r.message

    def test_same_title_and_year_is_possible_duplicate(self, library, monkeypatch):
        add(library, ask_llm=details_llm(title="Fox Behaviour", year="2019"))
        monkeypatch.setattr(lib, "read_file_properties",
                            lambda data, ext: {"title": "Fox behaviour", "year": "2019"})
        r = library.check(*item("other.pdf"))
        assert r.status == "possible_duplicate" and "title" in r.message.lower()

    def test_no_text(self, library):
        proc = processed("scan.pdf", [])
        proc["content"] = ""
        proc["error"] = "No text could be extracted from scan.pdf."
        r = library.check("scan.pdf", b"x", proc)
        assert (r.status, r.action) == ("no_text", "skip")

    def test_extraction_error(self, library):
        proc = processed("bad.pdf", [])
        proc["error"] = "Error processing document: boom"
        r = library.check("bad.pdf", b"x", proc)
        assert (r.status, r.action) == ("error", "skip")

    def test_duplicates_within_one_batch(self, library):
        results = library.check_many([item("a.pdf", data=b"same", prefix="a"),
                                      item("b.pdf", data=b"same", prefix="a")])
        assert [r.status for r in results] == ["new", "duplicate"]


# -----------------------------------------------------------------------------
class TestAddAndPersist:
    def test_add_sets_details_and_survives_reopen(self, library, root):
        rec = add(library, collection="Week 1", ask_llm=details_llm())
        assert rec.complete and rec.chunk_count == 3
        assert (rec.title, rec.title_source) == ("A Study of Foxes", "llm")
        assert rec.collection == "Week 1"
        path = library.path
        library.close()

        reopened = lib.Library.open(path, embedder=StubEmbedder())
        try:
            assert [d.doc_id for d in reopened.documents()] == [rec.doc_id]
            hits = reopened.store.search("zq3xsmith2021 marker")
            assert hits and hits[0].chunk.extra["doc_id"] == rec.doc_id
            assert reopened.store.sparse_search("zq3xsmith2021", k=3)
        finally:
            reopened.close()
        # Fixture's close() on the already-closed library must be harmless.
        library.close = lambda: None

    def test_add_without_llm_falls_back_to_file_properties(self, library, monkeypatch):
        monkeypatch.setattr(lib, "read_file_properties",
                            lambda data, ext: {"title": "Stored Title"})
        rec = add(library, ask_llm=silent_llm)
        assert (rec.title, rec.title_source) == ("Stored Title", "file_properties")

    def test_skip_action_is_refused(self, library):
        r = library.check(*item())
        r.action = "skip"
        with pytest.raises(ValueError):
            library.add(r, "General")

    def test_replace(self, library):
        old = add(library, ask_llm=details_llm(doi="10.1000/a"))
        name, data, proc = item("v2.pdf")
        proc["blocks"][0]["text"] += " 10.1000/a"
        proc["content"] = "\n\n".join(b["text"] for b in proc["blocks"])
        r = library.check(name, data, proc)
        assert r.status == "possible_duplicate" and r.existing.doc_id == old.doc_id
        r.action = "replace"
        new = library.add(r, "General")
        assert [d.doc_id for d in library.documents()] == [new.doc_id]
        assert library.db.get_document(old.doc_id) is None

    def test_keep_both(self, library):
        add(library, ask_llm=details_llm(doi="10.1000/a"))
        name, data, proc = item("v2.pdf")
        proc["blocks"][0]["text"] += " 10.1000/a"
        proc["content"] = "\n\n".join(b["text"] for b in proc["blocks"])
        r = library.check(name, data, proc)
        assert r.status == "possible_duplicate"
        r.action = "keep_both"
        library.add(r, "General")
        assert len(library.documents()) == 2

    def test_contextual_headers_when_enabled(self, library):
        library.info.contextual_headers = True
        rec = add(library, ask_llm=lambda p: "This excerpt is about foxes in the study.")
        chunks = library.db.load_chunks(rec.doc_id)
        assert all(c.extra.get("context_header") for c in chunks)

    def test_failed_add_leaves_nothing_behind(self, library, monkeypatch):
        r = library.check(*item())

        def boom(*a, **k):
            raise RuntimeError("embedding failed")
        monkeypatch.setattr(library.store, "add_chunks", boom)
        with pytest.raises(RuntimeError):
            library.add(r, "General")
        assert library.db.list_documents(include_incomplete=True) == []


# -----------------------------------------------------------------------------
class TestRecoveryAndMutation:
    def test_incomplete_document_removed_on_open(self, library):
        rec = add(library)
        # Simulate a crash mid-add: a second document written but never completed.
        name, data, proc = item("crashed.pdf")
        r = library.check(name, data, proc)
        from rag_core.chunker import chunk_document
        library.db.insert_document(DocumentRecord(
            doc_id="crash1", filename="crashed.pdf", file_size=1, doc_type=".pdf",
            page_or_slide_count=3, file_sha256=r.file_sha256, text_sha256=r.text_sha256,
            collection="General", added_at="2026-01-01T00:00:00", chunk_count=0, complete=False))
        chunks = chunk_document(proc["blocks"], "crashed.pdf", ".pdf", doc_key="crash1")
        library.db.insert_chunks("crash1", chunks)
        library.store.add_chunks(chunks)
        path = library.path
        library.close()

        reopened = lib.Library.open(path, embedder=StubEmbedder())
        try:
            assert reopened.recovered == ["crashed.pdf"]
            assert [d.doc_id for d in reopened.documents()] == [rec.doc_id]
            assert reopened.store._collection.count() == rec.chunk_count
        finally:
            reopened.close()
        library.close = lambda: None

    def test_remove(self, library):
        a = add(library, "a.pdf")
        b = add(library, "b.pdf")
        library.remove(a.doc_id)
        assert [d.doc_id for d in library.documents()] == [b.doc_id]
        assert library.store.sparse_search("zq3xa", k=5) == []
        assert library.store._collection.count() == b.chunk_count

    def test_move_updates_filters(self, library):
        rec = add(library, collection="A")
        library.move(rec.doc_id, "B")
        assert library.db.get_document(rec.doc_id).collection == "B"
        assert library.store.search("marker", where={"collection": "B"})
        assert library.store.search("marker", where={"collection": "A"}) == []
        assert all(c.extra["collection"] == "B" for c in library.db.load_chunks(rec.doc_id))

    def test_rename_collection(self, library):
        add(library, "a.pdf", collection="Old")
        add(library, "b.pdf", collection="Old")
        library.rename_collection("Old", "New")
        assert library.collections() == ["New"]
        assert len(library.store.search("marker", where={"collection": "New"})) >= 2

    def test_user_edit_marks_source(self, library):
        rec = add(library, ask_llm=details_llm())
        library.edit_details(rec.doc_id, title="Corrected", year="2020")
        got = library.db.get_document(rec.doc_id)
        assert (got.title, got.title_source, got.year, got.year_source) == \
            ("Corrected", "user", "2020", "user")

    def test_reembed(self, library):
        rec = add(library)
        library.reembed("stub-embed-2", embedder=StubEmbedder("stub-embed-2"))
        assert library.info.collection_name == "chunks_v2"
        assert library.info.embedding_model == "stub-embed-2"
        assert lib.read_info(library.path).collection_name == "chunks_v2"
        assert library.store._collection.count() == rec.chunk_count
        assert library.store.search("zq3xsmith2021 marker")

    def test_interrupted_reembed_keeps_old_collection(self, library):
        add(library)

        class Failing(StubEmbedder):
            def embed_documents(self, texts, progress_callback=None):
                raise RuntimeError("model crashed")

        with pytest.raises(RuntimeError):
            library.reembed("bad-model", embedder=Failing("bad-model"))
        assert library.info.collection_name == "chunks_v1"
        assert library.info.embedding_model == "stub-embed"
        assert library.store.search("zq3xsmith2021 marker")

    def test_disk_usage_positive(self, library):
        add(library)
        assert library.disk_usage() > 0
