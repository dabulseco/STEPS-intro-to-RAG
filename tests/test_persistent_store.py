"""Tests for the on-disk hybrid store."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library_fakes import StubEmbedder, make_blocks
from rag_core.chunker import chunk_document
from rag_core.persistent_store import PersistentRagStore, release_chroma


def doc_chunks(doc="d1", n=4, collection="General"):
    chunks = chunk_document(make_blocks(n, doc), f"{doc}.pdf", ".pdf", doc_key=doc)
    for c in chunks:
        c.extra["collection"] = collection
        c.extra["doc_id"] = doc
    return chunks


@pytest.fixture
def chroma_path(tmp_path):
    path = tmp_path / "chroma"
    yield path
    release_chroma()


class TestPersistence:
    def test_reopen_returns_same_results_without_reembedding(self, chroma_path):
        chunks = doc_chunks()
        first = PersistentRagStore(StubEmbedder(), chroma_path)
        first.add_chunks(chunks)
        before = [h.chunk.chunk_id for h in first.search("zq3xd1 marker")]
        first.close()

        class NoEmbed(StubEmbedder):
            def embed_documents(self, texts, progress_callback=None):
                raise AssertionError("reopening must not re-embed documents")

        second = PersistentRagStore(NoEmbed(), chroma_path)
        second.load_chunks(chunks)
        hits = second.search("zq3xd1 marker")
        assert [h.chunk.chunk_id for h in hits] == before
        assert any(h.dense_rank is not None for h in hits)
        assert any(h.sparse_rank is not None for h in hits)
        second.close()

    def test_close_keeps_vectors_on_disk(self, chroma_path):
        store = PersistentRagStore(StubEmbedder(), chroma_path)
        store.add_chunks(doc_chunks())
        store.close()
        again = PersistentRagStore(StubEmbedder(), chroma_path)
        assert again._collection.count() == 4
        again.close()

    def test_missing_vector_ids(self, chroma_path):
        chunks = doc_chunks()
        store = PersistentRagStore(StubEmbedder(), chroma_path)
        store.add_chunks(chunks[:2])
        missing = store.missing_vector_ids([c.chunk_id for c in chunks])
        assert missing == [c.chunk_id for c in chunks[2:]]
        store.close()


class TestMutation:
    def test_remove_chunks_from_both_indexes(self, chroma_path):
        keep, drop = doc_chunks("d1"), doc_chunks("d2")
        store = PersistentRagStore(StubEmbedder(), chroma_path)
        store.add_chunks(keep + drop)
        store.remove_chunks([c.chunk_id for c in drop])

        assert store.count == len(keep)
        assert store._collection.count() == len(keep)
        assert store.sparse_search("zq3xd2", k=5) == []
        ids = {h.chunk.chunk_id for h in store.search("zq3xd2 marker")}
        assert ids.isdisjoint({c.chunk_id for c in drop})
        store.close()

    def test_update_chunk_metadata_changes_filter_results(self, chroma_path):
        chunks = doc_chunks(collection="A")
        store = PersistentRagStore(StubEmbedder(), chroma_path)
        store.add_chunks(chunks)
        assert store.search("marker", where={"collection": "B"}) == []

        for c in chunks:
            c.extra["collection"] = "B"
        store.update_chunk_metadata(chunks)
        assert store.search("marker", where={"collection": "B"})
        assert store.search("marker", where={"collection": "A"}) == []
        assert store.get_chunk(chunks[0].chunk_id).extra["collection"] == "B"
        store.close()

    def test_separate_collection_names_are_independent(self, chroma_path):
        v1 = PersistentRagStore(StubEmbedder(), chroma_path, collection_name="chunks_v1")
        v1.add_chunks(doc_chunks())
        v2 = PersistentRagStore(StubEmbedder(), chroma_path, collection_name="chunks_v2")
        assert v2._collection.count() == 0
        v1.close(); v2.close()
