"""
Smoke tests for the Streamlit library app.

Runs the real script with Streamlit's AppTest harness. Ollama is replaced by
stubs: model lists are patched, embeddings use StubEmbedder, and the chat
model is a plain function.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from streamlit.testing.v1 import AppTest

from library_fakes import StubEmbedder, make_blocks, processed
from library_ui import state
from rag_core import library as lib
from rag_core.conversations import ConversationStore
from rag_core.persistent_store import release_chroma

APP = str(ROOT / "library_app.py")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(lib.LIBRARY_ROOT_ENV, str(tmp_path / "libs"))
    monkeypatch.setattr(lib, "OllamaEmbedder", lambda model: StubEmbedder(model))
    monkeypatch.setattr(state, "ollama_status", lambda: state.OllamaStatus(
        chat_models=["stub-chat"], embedding_models=["stub-embed"]))

    def fake_llms(settings):
        chat = lambda messages: "Foxes jump over the dog [S1]."
        ask = lambda prompt: "1. Why do foxes jump?\n2. How high?\n3. How often?"
        return chat, ask
    monkeypatch.setattr(state, "make_llms", fake_llms)
    yield tmp_path / "libs"
    state.close_all()
    release_chroma()


def run(at=None):
    at = at or AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def seeded_library(chat_model="stub-chat"):
    path = lib.create_library("BIOL 201", "stub-embed")
    library = lib.Library.open(path, embedder=StubEmbedder())
    library.info.chatbot.chat_model = chat_model
    library.info.chatbot.rerank = False
    library.info.chatbot.condense_history = False
    library.save_info()
    check = library.check("smith2021.pdf", b"smith", processed("smith2021.pdf", make_blocks(3, "smith")))
    library.add(check, "Week 1")
    library.close()
    return path


def texts(at):
    return " ".join(str(e.value) for e in list(at.markdown) + list(at.header) + list(at.subheader)
                    + list(at.caption) + list(at.info) + list(at.warning) + list(at.error))


class TestApp:
    def test_first_run_shows_create_form(self, env):
        at = run()
        assert any("Create a library" in h.value for h in at.header)

    def test_create_library_from_form(self, env):
        at = run()
        at.text_input[0].input("Cell Bio")
        next(b for b in at.button if b.label == "Create library").click()
        at = run(at)
        assert [i.name for _, i in lib.list_libraries()] == ["Cell Bio"]
        assert at.sidebar.selectbox[0].value == str(env / "cell-bio")

    def test_documents_tab_lists_documents(self, env):
        seeded_library()
        at = run()
        at.sidebar.radio(key="tab").set_value("📄 Documents")
        at = run(at)
        assert at.dataframe, "document table missing"
        assert "In this library (1)" in texts(at)

    def test_ask_saves_conversation_and_reloads(self, env):
        path = seeded_library()
        at = run()
        at.chat_input[0].set_value("What does the smith marker show?")
        at = run(at)
        page = texts(at)
        assert "Foxes jump over the dog [S1]." in page

        library = state.get_library(path)
        [conv] = ConversationStore(library.db).list()
        [ex] = ConversationStore(library.db).exchanges(conv.conversation_id)
        assert ex.sources and ex.followups == ["Why do foxes jump?", "How high?", "How often?"]

        # A fresh session (as after restarting the app) can reopen the chat.
        state.close_all()
        at2 = run()
        at2.sidebar.button(key=f"conv_{conv.conversation_id}").click()
        at2 = run(at2)
        assert "Foxes jump over the dog [S1]." in texts(at2)

    def test_missing_chat_model_prompts_for_choice(self, env):
        seeded_library(chat_model="gone-model")
        at = run()
        assert "is not installed" in texts(at)
        assert at.chat_input[0].disabled

    def test_missing_embedding_model_is_reported(self, env, monkeypatch):
        seeded_library()
        monkeypatch.setattr(state, "ollama_status", lambda: state.OllamaStatus(
            chat_models=["stub-chat"], embedding_models=["other-embed"]))
        at = run()
        assert "ollama pull stub-embed" in texts(at)

    def test_ollama_down_still_browses(self, env, monkeypatch):
        seeded_library()
        monkeypatch.setattr(state, "ollama_status",
                            lambda: state.OllamaStatus(error="connection refused"))
        at = run()
        assert "Ollama is not reachable" in texts(at)
        at.sidebar.radio(key="tab").set_value("📄 Documents")
        at = run(at)
        assert at.dataframe

    def test_settings_tabs_render(self, env):
        seeded_library()
        at = run()
        for tab in ("⚙️ Chatbot settings", "🗄️ Library settings"):
            at.sidebar.radio(key="tab").set_value(tab)
            at = run(at)
