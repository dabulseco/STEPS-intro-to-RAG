"""Tests for chat orchestration over a persistent library."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library_fakes import RecordingChat, StubEmbedder, make_blocks, processed
from rag_core import chat_session as cs
from rag_core import library as lib
from rag_core.conversations import ConversationStore
from rag_core.persistent_store import release_chroma
from utils.prompt_builder import PromptBuilder


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setenv(lib.LIBRARY_ROOT_ENV, str(tmp_path / "libs"))
    path = lib.create_library("BIOL 201", "stub-embed")
    opened = lib.Library.open(path, embedder=StubEmbedder())
    # Keep the tests about orchestration, not about query rewriting.
    opened.info.chatbot.condense_history = False
    opened.info.chatbot.rerank = False
    yield opened
    opened.close()
    release_chroma()


def add_doc(library, name, collection="General"):
    check = library.check(name, name.encode(), processed(name, make_blocks(3, name.split(".")[0])))
    return library.add(check, collection)


def no_llm(prompt):
    return ""


class TestSystemPrompt:
    def test_custom_instructions_appended_not_replacing(self):
        settings = lib.ChatbotSettings(custom_instructions="Explain terms before using them.")
        base = PromptBuilder.build_system_prompt("Scientist", "Scientist", True, "strict")
        prompt = cs.build_system_prompt(settings, True, "strict")
        assert prompt.startswith(base)
        assert prompt.rstrip().endswith("Explain terms before using them.")

    def test_empty_instructions_add_nothing(self):
        settings = lib.ChatbotSettings(custom_instructions="   ")
        base = PromptBuilder.build_system_prompt("Scientist", "Scientist", True, "strict")
        assert cs.build_system_prompt(settings, True, "strict") == base


class TestCollectionFilter:
    def test_all_or_none_selected_means_no_filter(self):
        s = lib.ChatbotSettings(collections=[])
        assert cs.collection_filter(s, ["A", "B"]) is None
        s.collections = ["A", "B"]
        assert cs.collection_filter(s, ["A", "B"]) is None

    def test_subset(self):
        s = lib.ChatbotSettings(collections=["A", "Gone"])
        assert cs.collection_filter(s, ["A", "B"]) == {"collection": {"$in": ["A"]}}

    def test_only_deleted_collections_means_no_filter(self):
        s = lib.ChatbotSettings(collections=["Gone"])
        assert cs.collection_filter(s, ["A"]) is None


class TestFollowups:
    def test_parsed(self):
        ask = lambda p: "1. Why?\n2) How?\n3. When?\n4. Extra?"
        assert cs.generate_followups("q", "a", ask) == ["Why?", "How?", "When?"]

    def test_failure_gives_empty(self):
        def boom(p):
            raise RuntimeError("down")
        assert cs.generate_followups("q", "a", boom) == []


class TestAsk:
    def test_answer_with_sources_and_history_window(self, library):
        add_doc(library, "smith2021.pdf")
        chat = RecordingChat("Foxes jump [S1].")
        history = [(f"q{i}", f"a{i}") for i in range(8)]
        result = cs.ask(library, "What does zq3xsmith2021 mark?", history, chat, no_llm)

        assert result.rag.answer == "Foxes jump [S1]."
        assert result.rag.sources
        assert result.rag.citations.is_clean
        messages = chat.calls[0]
        assert messages[0][0] == "system"
        assert [m[1] for m in messages[1:-1:2]] == [f"q{i}" for i in range(3, 8)]
        assert messages[-1][0] == "user" and "DOCUMENT EXCERPTS" in messages[-1][1]

    def test_saved_sources_resolve_doc_and_survive_removal(self, library):
        rec = add_doc(library, "smith2021.pdf")
        result = cs.ask(library, "zq3xsmith2021 marker", [], RecordingChat(), no_llm)
        saved = cs.saved_sources(library, result.rag.sources)
        assert saved[0].marker == "[S1]" and saved[0].doc_id == rec.doc_id
        assert saved[0].passage and len(saved[0].passage) <= 600

        conversations = ConversationStore(library.db)
        cid = conversations.create()
        conversations.append(cid, "q", result.rag.answer, "m", saved, result.rag.citations, [])
        library.remove(rec.doc_id)
        [ex] = conversations.exchanges(cid)
        assert ex.sources[0].passage == saved[0].passage
        assert cs.saved_sources(library, result.rag.sources)[0].doc_id == ""

    def test_collection_filter_limits_sources(self, library):
        add_doc(library, "a.pdf", collection="A")
        add_doc(library, "b.pdf", collection="B")
        library.info.chatbot.collections = ["B"]
        result = cs.ask(library, "marker study", [], RecordingChat(), no_llm)
        assert result.rag.sources
        assert {s.source_file for s in result.rag.sources} == {"b.pdf"}

    def test_strict_mode_empty_library_declines(self, library):
        chat = RecordingChat()
        result = cs.ask(library, "anything", [], chat, no_llm)
        assert chat.calls == []
        assert "No documents are indexed" in result.rag.answer

    def test_followups_generated_when_enabled(self, library):
        add_doc(library, "smith2021.pdf")
        result = cs.ask(library, "marker", [], RecordingChat(), lambda p: "1. A?\n2. B?\n3. C?")
        assert result.followups == ["A?", "B?", "C?"]
        library.info.chatbot.generate_followups = False
        result = cs.ask(library, "marker", [], RecordingChat(), lambda p: "1. A?")
        assert result.followups == []

    def test_rerank_on_with_unusable_llm_still_answers(self, library):
        add_doc(library, "smith2021.pdf")
        library.info.chatbot.rerank = True
        result = cs.ask(library, "zq3xsmith2021 marker", [], RecordingChat(), no_llm)
        assert result.rag.sources and result.rag.diagnostics.reranked
