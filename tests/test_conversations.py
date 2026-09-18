"""Tests for saved conversations."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core.citations import verify_citations
from rag_core.context_assembler import Source
from rag_core.conversations import ConversationStore, SavedSource
from rag_core.library_db import LibraryDB


@pytest.fixture
def store(tmp_path):
    db = LibraryDB(tmp_path / "library.db")
    yield ConversationStore(db)
    db.close()


def saved(n=1, text="Foxes jump over dogs in the study."):
    return [SavedSource(marker=f"[S{i}]", doc_id="d1", location_label=f"a.pdf p.{i}",
                        passage=text) for i in range(1, n + 1)]


def citations(answer, n=1):
    sources = [Source(index=i, chunk_id=f"c{i}", text="", location_label="",
                      source_file="a.pdf") for i in range(1, n + 1)]
    return verify_citations(answer, sources)


class TestConversations:
    def test_create_and_list_newest_first(self, store):
        a = store.create()
        b = store.create("Second")
        assert [c.conversation_id for c in store.list()] == [b, a]
        store.append(a, "q", "a", "llama3", [], None, [])
        assert store.list()[0].conversation_id == a
        assert store.list()[0].exchanges == 1

    def test_append_and_read_back(self, store):
        cid = store.create()
        answer = "Foxes jump [S1]. They also run [S9]."
        ex = store.append(cid, "What do foxes do?", answer, "llama3",
                          saved(1, "x" * 900), citations(answer), ["Why?", "How?"])
        assert ex.position == 0
        [back] = store.exchanges(cid)
        assert back.question == "What do foxes do?" and back.answer == answer
        assert back.chat_model == "llama3"
        assert len(back.sources[0].passage) == 600
        assert back.sources[0].marker == "[S1]" and back.sources[0].doc_id == "d1"
        assert back.citation_clean is False and back.hallucinated == [9]
        assert "[S9]" in back.citation_summary
        assert back.followups == ["Why?", "How?"]

    def test_no_citation_report(self, store):
        cid = store.create()
        store.append(cid, "q", "a", "m", [], None, [])
        [ex] = store.exchanges(cid)
        assert ex.citation_clean is True and ex.citation_summary == ""

    def test_title_from_first_question(self, store):
        cid = store.create()
        store.append(cid, "  How do red foxes " + "hunt small rodents " * 5, "a", "m", [], None, [])
        title = store.list()[0].title
        assert title.startswith("How do red foxes") and len(title) <= 61
        store.append(cid, "second question", "a", "m", [], None, [])
        assert store.list()[0].title == title

    def test_explicit_title_kept(self, store):
        cid = store.create("Quiz prep")
        store.append(cid, "first", "a", "m", [], None, [])
        assert store.list()[0].title == "Quiz prep"

    def test_history_window(self, store):
        cid = store.create()
        for i in range(7):
            store.append(cid, f"q{i}", f"a{i}", "m", [], None, [])
        assert store.history(cid) == [(f"q{i}", f"a{i}") for i in range(2, 7)]
        assert store.history(cid, window=2) == [("q5", "a5"), ("q6", "a6")]

    def test_rename_and_delete(self, store):
        cid = store.create()
        store.append(cid, "q", "a", "m", saved(), None, [])
        store.rename(cid, "Renamed")
        assert store.list()[0].title == "Renamed"
        store.delete(cid)
        assert store.list() == []
        assert store.exchanges(cid) == []

    def test_survives_reopen(self, tmp_path):
        db = LibraryDB(tmp_path / "library.db")
        cid = ConversationStore(db).create()
        ConversationStore(db).append(cid, "q", "a", "m", saved(), None, ["f"])
        db.close()
        db2 = LibraryDB(tmp_path / "library.db")
        [ex] = ConversationStore(db2).exchanges(cid)
        assert ex.followups == ["f"] and ex.sources[0].passage
        db2.close()
