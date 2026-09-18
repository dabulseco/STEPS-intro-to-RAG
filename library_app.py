"""
Persistent Library App.

Keeps named document libraries on disk and serves each one as a configurable
chatbot with saved conversations. The original app (`app.py`) is unchanged
and still forgets everything when closed.

Run with:  streamlit run library_app.py
Libraries live in ./libraries (override with the RAG_LIBRARY_DIR variable).
"""

from pathlib import Path

import streamlit as st

from library_ui import chat_tab, documents_tab, settings_tabs, state
from rag_core.library import list_libraries, library_root
from rag_core.library_lock import LibraryLocked

st.set_page_config(page_title="RAG Library", page_icon="📚", layout="wide")

TABS = ["💬 Chat", "📄 Documents", "⚙️ Chatbot settings", "🗄️ Library settings"]

status = state.ollama_status()

# -----------------------------------------------------------------------------
# Sidebar: library picker
# -----------------------------------------------------------------------------
st.sidebar.title("📚 RAG Library")
libraries = list_libraries()
paths = [str(p) for p, _ in libraries]
names = {str(p): info.name for p, info in libraries}

if st.session_state.get("library_path") not in paths:
    st.session_state.library_path = paths[0] if paths else None

creating = st.session_state.get("creating", False) or not libraries

if libraries:
    chosen = st.sidebar.selectbox(
        "Library", paths, index=paths.index(st.session_state.library_path),
        format_func=names.get,
    )
    if chosen != st.session_state.library_path:
        st.session_state.library_path = chosen
        st.session_state.active_conversation = None
        st.session_state.creating = False
        st.rerun()
if st.sidebar.button("➕ New library", use_container_width=True):
    st.session_state.creating = True
    st.rerun()

if not status.available:
    st.error(f"⚠️ Ollama is not reachable: {status.error}. Start Ollama "
             "(`ollama serve`) to add documents or ask questions. "
             "You can still browse libraries and conversations.")

# -----------------------------------------------------------------------------
# Create a library
# -----------------------------------------------------------------------------
if creating:
    if libraries and st.button("← Back"):
        st.session_state.creating = False
        st.rerun()
    new_path = settings_tabs.render_create_form(status)
    if new_path is not None:
        st.session_state.library_path = str(new_path)
        st.session_state.creating = False
        st.session_state.active_conversation = None
        st.rerun()
    if not libraries:
        st.caption(f"Libraries are stored in `{library_root()}`.")
    st.stop()

# -----------------------------------------------------------------------------
# Open the selected library
# -----------------------------------------------------------------------------
try:
    library = state.get_library(Path(st.session_state.library_path))
except LibraryLocked as e:
    st.error(f"🔒 {e}")
    st.stop()
except Exception as e:  # noqa: BLE001 -- UI boundary
    st.error(f"Could not open this library: {e}")
    st.stop()

if library.recovered:
    st.warning(
        "The app was closed while these documents were being added, so they were "
        "removed and need adding again: " + ", ".join(library.recovered)
    )
    library.recovered = []

info = library.info
embed_ok = status.available and state.model_installed(info.embedding_model, status.embedding_models)
if status.available and not embed_ok:
    st.error(
        f"This library's embedding model `{info.embedding_model}` is not installed, "
        f"so it cannot be searched or added to. Install it with:\n\n"
        f"`ollama pull {info.embedding_model}`"
    )

chat_ok = status.available and state.model_installed(info.chatbot.chat_model, status.chat_models)
ask_llm = state.make_llms(info.chatbot)[1] if chat_ok else None

tab = st.sidebar.radio("View", TABS, key="tab")
if tab == TABS[0]:
    chat_tab.render_sidebar(library)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
if tab == TABS[0]:
    chat_tab.render(library, status, ready=embed_ok)

elif tab == TABS[1]:
    add_ready = embed_ok
    if embed_ok and not chat_ok:
        if info.contextual_headers:
            st.warning("This library adds contextual headers, which needs a chat "
                       "model. Choose one in Chatbot settings before adding documents.")
            add_ready = False
        else:
            st.info("No chat model is set, so titles and authors will come only from "
                    "the files' stored properties. Choose a chat model in Chatbot "
                    "settings for better details.")
    documents_tab.render(library, status, add_ready, ask_llm)

elif tab == TABS[2]:
    settings_tabs.render_chatbot_settings(library, status)

else:
    if settings_tabs.render_library_settings(library, status):
        st.session_state.library_path = None
        st.session_state.active_conversation = None
        st.rerun()
