"""Create-library form, Chatbot settings tab and Library settings tab."""

from pathlib import Path
from typing import Optional

import streamlit as st

from rag_core.library import Library, create_library, delete_library
from rag_core.library_lock import LibraryLocked
from utils.prompt_builder import PromptBuilder

from . import state

DEFAULT_EMBED_MODEL = "nomic-embed-text"


# -----------------------------------------------------------------------------
# Create library
# -----------------------------------------------------------------------------
def render_create_form(status: state.OllamaStatus) -> Optional[Path]:
    """The 'Create library' form. Returns the new library's path once created."""
    st.header("📚 Create a library")
    st.write(
        "A library keeps your documents' text and search index on this computer, "
        "so you can close the app and come back to the same chatbot and "
        "conversations. The original files are not kept."
    )
    if not status.available:
        st.warning(f"Ollama is not reachable ({status.error}). You can create a "
                   "library now, but adding documents needs Ollama running.")

    with st.form("create_library"):
        name = st.text_input("Library name", placeholder="e.g. BIOL 201 readings")
        if status.embedding_models:
            options = status.embedding_models
            default = next((i for i, m in enumerate(options)
                            if state.model_installed(DEFAULT_EMBED_MODEL, [m])), 0)
            embed_model = st.selectbox(
                "Embedding model", options, index=default,
                help="Fixed for this library. Changing it later means re-embedding "
                     "every document (Library settings → Re-embed).",
            )
        else:
            embed_model = st.text_input(
                "Embedding model", value=DEFAULT_EMBED_MODEL,
                help="No embedding models found. Install one with "
                     f"`ollama pull {DEFAULT_EMBED_MODEL}`.",
            )
        headers = st.checkbox(
            "Add contextual headers to chunks (slow)",
            value=False,
            help="Uses the chat model to add a situating sentence to every chunk "
                 "before embedding, which improves retrieval. Costs one model call "
                 "per chunk — roughly 75 calls for a 15-page paper. Applies to every "
                 "document in this library and cannot be changed later.",
        )
        submitted = st.form_submit_button("Create library", type="primary")

    if submitted:
        if not name.strip():
            st.error("Give the library a name.")
            return None
        if not embed_model.strip():
            st.error("Choose an embedding model.")
            return None
        return create_library(name.strip(), embed_model.strip(), contextual_headers=headers)
    return None


# -----------------------------------------------------------------------------
# Chatbot settings
# -----------------------------------------------------------------------------
def render_chatbot_settings(library: Library, status: state.OllamaStatus) -> None:
    s = library.info.chatbot
    st.header("⚙️ Chatbot settings")
    st.caption("Saved with this library and used for every conversation in it.")

    perspectives = PromptBuilder.get_available_perspectives()
    audiences = PromptBuilder.get_available_audiences()
    chat_models = list(status.chat_models)
    if s.chat_model and s.chat_model not in chat_models:
        chat_models.insert(0, s.chat_model)

    with st.form("chatbot_settings"):
        st.subheader("Identity")
        name = st.text_input("Chatbot name", value=s.name)
        description = st.text_input("Description", value=s.description)
        custom = st.text_area(
            "Custom instructions", value=s.custom_instructions, height=120,
            help="Added to the app's built-in rules. They cannot switch off answering "
                 "from the documents or the [S1] citation rules.",
            placeholder="e.g. You are a tutor for introductory cell biology. "
                        "Explain terms before using them.",
        )

        st.subheader("Voice")
        c1, c2 = st.columns(2)
        perspective = c1.selectbox("Perspective (who is writing)", perspectives,
                                   index=_index(perspectives, s.perspective))
        audience = c2.selectbox("Audience (who is reading)", audiences,
                                index=_index(audiences, s.audience))

        st.subheader("Model")
        if chat_models:
            chat_model = st.selectbox("Chat model", chat_models,
                                      index=_index(chat_models, s.chat_model))
        else:
            chat_model = st.text_input("Chat model", value=s.chat_model,
                                       help="Ollama is not reachable; type the model name.")
        c1, c2 = st.columns(2)
        temperature = c1.slider("Temperature", 0.0, 2.0, float(s.temperature), 0.05)
        top_k = c2.slider("Top-K", 1, 100, int(s.top_k), 1)
        top_p = c1.slider("Top-P", 0.0, 1.0, float(s.top_p), 0.01)
        num_ctx = c2.slider("Context length", 256, 32768, int(s.num_ctx), 256)

        st.subheader("Answering")
        allow_llm = st.checkbox(
            "Allow LLM general knowledge", value=s.allow_llm_knowledge,
            help="Off: answers come only from this library's documents.",
        )
        followups = st.checkbox("Generate follow-up questions", value=s.generate_followups)

        st.subheader("Retrieval")
        collections = library.collections()
        chosen = st.multiselect(
            "Collections searched", collections,
            default=[c for c in s.collections if c in collections],
            help="Leave empty to search every collection.",
        )
        dense_weight = st.slider("Semantic ← → Keyword balance", 0.0, 1.0,
                                 float(s.dense_weight), 0.05,
                                 help="1.0 = semantic only, 0.0 = keyword only.")
        c1, c2, c3 = st.columns(3)
        final_k = c1.slider("Passages sent to the model", 1, 15, int(s.final_k), 1)
        fused_k = c2.slider("Candidates considered", 1, 40, int(s.fused_k), 1)
        dense_k = c3.slider("Candidates per index", 5, 50, int(s.dense_k), 5)
        c1, c2 = st.columns(2)
        condense = c1.checkbox("Resolve follow-up references", value=s.condense_history)
        rewrite = c2.checkbox("Rewrite query for retrieval", value=s.rewrite_query)
        multi = c1.checkbox("Search multiple phrasings", value=s.multi_query)
        hyde = c2.checkbox("HyDE (search by hypothetical answer)", value=s.hyde)
        infer = c1.checkbox("Infer document/section filter", value=s.infer_filter)
        rerank = c2.checkbox("Re-rank retrieved passages", value=s.rerank,
                             help="Recommended for larger libraries; adds time per answer.")
        backends = ["llm", "cross-encoder"]
        backend = st.selectbox(
            "Re-ranker", backends, index=_index(backends, s.rerank_backend),
            format_func=lambda b: "LLM (no extra install)" if b == "llm"
            else "Cross-encoder (needs sentence-transformers)",
        )

        saved = st.form_submit_button("Save settings", type="primary")

    if saved:
        s.name = name.strip() or library.info.name
        s.description = description.strip()
        s.custom_instructions = custom.strip()
        s.perspective, s.audience = perspective, audience
        s.chat_model = (chat_model or "").strip()
        s.temperature, s.top_k, s.top_p, s.num_ctx = temperature, top_k, top_p, num_ctx
        s.allow_llm_knowledge, s.generate_followups = allow_llm, followups
        s.collections = chosen
        s.dense_weight, s.final_k, s.dense_k = dense_weight, final_k, dense_k
        s.fused_k = max(fused_k, final_k)
        s.condense_history, s.rewrite_query, s.multi_query = condense, rewrite, multi
        s.hyde, s.infer_filter, s.rerank, s.rerank_backend = hyde, infer, rerank, backend
        library.save_info()
        st.success("Settings saved.")


def _index(options, value) -> int:
    try:
        return list(options).index(value)
    except ValueError:
        return 0


# -----------------------------------------------------------------------------
# Library settings
# -----------------------------------------------------------------------------
def render_library_settings(library: Library, status: state.OllamaStatus) -> bool:
    """Returns True when the library was deleted (caller resets the selection)."""
    info = library.info
    st.header("🗄️ Library settings")

    with st.form("rename_library"):
        new_name = st.text_input("Library name", value=info.name)
        if st.form_submit_button("Rename") and new_name.strip() and new_name.strip() != info.name:
            library.rename(new_name)
            st.success("Renamed.")
            st.rerun()

    st.subheader("Storage")
    st.write(f"**Folder:** `{library.path}`")
    st.write(f"**Size on disk:** {state.format_bytes(library.disk_usage())} · "
             f"**Documents:** {len(library.documents())} · "
             f"**Chunks:** {library.store.count:,}")
    st.write(f"**Embedding model:** `{info.embedding_model}` · "
             f"**Contextual headers:** {'on' if info.contextual_headers else 'off'}")
    st.caption("To back up this library, copy its folder while the app is closed.")

    st.subheader("Re-embed with a different model")
    st.write(
        "Re-computes every chunk's vector with a new embedding model. No re-upload "
        "is needed, but it takes about as long as adding the documents did. If it "
        "is interrupted, the library keeps using the current model."
    )
    others = [m for m in status.embedding_models
              if not state.model_installed(info.embedding_model, [m])]
    if not status.available:
        st.info("Ollama is not reachable, so re-embedding is unavailable.")
    elif not others:
        st.info("No other embedding models are installed.")
    else:
        model = st.selectbox("New embedding model", others)
        if st.button(f"Re-embed {library.store.count:,} chunks with {model}"):
            bar = st.progress(0.0, text="Re-embedding…")
            try:
                library.reembed(model, progress=lambda stage, d, t: bar.progress(
                    d / t if t else 1.0, text=f"Re-embedding… {d:,}/{t:,}"))
            except Exception as e:  # noqa: BLE001 -- UI boundary
                st.error(f"Re-embedding failed; the library still uses "
                         f"{info.embedding_model}. ({e})")
            else:
                st.success(f"Done. The library now uses {model}.")
            finally:
                bar.empty()

    st.subheader("Delete library")
    st.write("Deletes the library folder: its index, document list and every "
             "conversation. This cannot be undone.")
    confirm = st.text_input(f"Type **{info.name}** to confirm", key="delete_confirm")
    if st.button("Delete library", type="primary", disabled=confirm.strip() != info.name):
        path = library.path
        state.forget_library(path)
        try:
            delete_library(path)
        except LibraryLocked as e:
            st.error(str(e))
            return False
        return True
    return False
