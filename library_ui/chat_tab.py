"""Chat tab and the conversation list in the sidebar."""

import datetime
import re
from typing import List, Optional, Sequence, Tuple

import streamlit as st

from rag_core.chat_session import ask, saved_sources
from rag_core.conversations import ConversationStore, Exchange
from rag_core.library import Library
from utils.html_export import build_conversation_html, build_pair_html, safe_filename

from . import settings_tabs, state


def render_sidebar(library: Library) -> None:
    """Conversation list (newest first) with a New chat button."""
    conversations = ConversationStore(library.db)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Conversations")
    if st.sidebar.button("➕ New chat", use_container_width=True):
        st.session_state.active_conversation = None
        st.rerun()
    active = st.session_state.get("active_conversation")
    for conv in conversations.list():
        label = ("▶ " if conv.conversation_id == active else "") + conv.title
        if st.sidebar.button(label, key=f"conv_{conv.conversation_id}",
                             use_container_width=True):
            st.session_state.active_conversation = conv.conversation_id
            st.rerun()


def render(library: Library, status: state.OllamaStatus, ready: bool) -> None:
    settings = library.info.chatbot
    conversations = ConversationStore(library.db)
    active: Optional[int] = st.session_state.get("active_conversation")
    if active is not None and active not in {c.conversation_id for c in conversations.list()}:
        active = st.session_state.active_conversation = None

    st.header(f"💬 {settings.name or library.info.name}")
    scope = ", ".join(settings.collections) if settings.collections else "all collections"
    st.caption(f"{settings.chat_model or 'no chat model'} · "
               f"{settings.perspective} → {settings.audience} · {scope}")
    if settings.description:
        st.write(settings.description)

    model_ok = _check_chat_model(library, status)

    exchanges = conversations.exchanges(active) if active is not None else []
    if active is not None:
        _render_manage(library, conversations, active, exchanges, settings)
    for exchange in exchanges:
        _render_exchange(exchange)
    if exchanges:
        # Where people look once they have finished asking: below the last answer.
        html_doc, filename = conversation_export(library.info.name, exchanges, settings)
        st.download_button(
            "⬇️ Download full conversation (HTML)", html_doc.encode("utf-8"),
            file_name=filename, mime="text/html", key="dl_full_conversation",
            type="primary",
        )

    pending = st.session_state.pop("pending_question", None)
    can_ask = ready and model_ok
    typed = st.chat_input("Ask about this library's documents…", disabled=not can_ask)
    question = typed or pending
    if question and can_ask:
        _answer(library, conversations, active, question)


def _check_chat_model(library: Library, status: state.OllamaStatus) -> bool:
    settings = library.info.chatbot
    if not status.available:
        return False
    if state.model_installed(settings.chat_model, status.chat_models):
        return True
    if settings.chat_model and status.chat_models:
        st.warning(f"This chatbot's model `{settings.chat_model}` is not installed. "
                   "Pick another to continue.")
    elif status.chat_models:
        st.info("Choose a chat model for this chatbot.")
    settings_tabs.render_chat_model_picker(library, status, key="chat_chat_model")
    return False


def _render_manage(library: Library, conversations: ConversationStore, active: int,
                   exchanges, settings) -> None:
    summary = next(c for c in conversations.list() if c.conversation_id == active)
    with st.expander(f"Manage “{summary.title}”"):
        with st.form(f"rename_conv_{active}"):
            title = st.text_input("Title", value=summary.title)
            if st.form_submit_button("Rename"):
                conversations.rename(active, title)
                st.rerun()
        if exchanges:
            html_doc, filename = conversation_export(library.info.name, exchanges, settings)
            st.download_button("⬇️ Download conversation (HTML)", html_doc.encode("utf-8"),
                               file_name=filename, mime="text/html")
        sure = st.checkbox("I want to delete this conversation", key=f"del_ok_{active}")
        if st.button("Delete conversation", disabled=not sure):
            conversations.delete(active)
            st.session_state.active_conversation = None
            st.rerun()


def conversation_export(
    library_name: str,
    exchanges: Sequence[Exchange],
    settings,
    now: Optional[datetime.datetime] = None,
) -> Tuple[str, str]:
    """
    (html, filename) for a full-conversation download.

    Named "<library> — <date time>" rather than after the first question, with
    the chat model(s) that answered listed directly below the title.
    """
    now = now or datetime.datetime.now()
    title = f"{library_name} — {now:%Y-%m-%d %H:%M}"
    stem = re.sub(r"\s+", "-", library_name.strip()) or "conversation"
    filename = safe_filename(f"{stem}_{now:%Y-%m-%d_%H%M}")
    models: List[str] = []
    for e in exchanges:
        if e.chat_model and e.chat_model not in models:
            models.append(e.chat_model)
    html_doc = build_conversation_html(
        [(e.question, e.answer) for e in exchanges], title=title,
        perspective=settings.perspective, audience=settings.audience, models=models,
    )
    return html_doc, filename


def _render_exchange(exchange: Exchange) -> None:
    with st.chat_message("user"):
        st.markdown(exchange.question)
    with st.chat_message("assistant"):
        st.markdown(exchange.answer)
        st.caption(f"{exchange.chat_model} · {exchange.created_at[:16].replace('T', ' ')}")
        if not exchange.citation_clean:
            st.warning(exchange.citation_summary)

        if exchange.sources:
            with st.expander(f"📚 Sources ({len(exchange.sources)})"):
                if exchange.citation_summary:
                    st.caption(exchange.citation_summary)
                for source in exchange.sources:
                    st.markdown(f"**{source.marker}** {source.location_label}")
                    st.text(source.passage)

        diagnostics = st.session_state.get("diagnostics", {}).get(exchange.message_id)
        if diagnostics is not None:
            with st.expander("🔬 Retrieval diagnostics (this session only)"):
                if diagnostics.search_query != diagnostics.original_query:
                    st.markdown(f"**Search query:** {diagnostics.search_query}")
                if diagnostics.where_filter:
                    st.markdown(f"**Filter:** `{diagnostics.where_filter}`")
                c1, c2, c3 = st.columns(3)
                c1.metric("Semantic hits", diagnostics.dense_hits)
                c2.metric("Keyword hits", diagnostics.sparse_hits)
                c3.metric("Sources used", diagnostics.sources_used)
                for note in diagnostics.notes:
                    st.info(note)

        c1, c2 = st.columns([4, 1])
        with c1:
            for j, followup in enumerate(exchange.followups):
                if st.button(f"❓ {followup}", key=f"fu_{exchange.message_id}_{j}"):
                    st.session_state.pending_question = followup
                    st.rerun()
        with c2:
            pair = build_pair_html(exchange.position, exchange.question, exchange.answer)
            st.download_button("⬇️ Q&A", pair.encode("utf-8"),
                               file_name=safe_filename(f"qa_{exchange.position + 1}"),
                               mime="text/html", key=f"dl_{exchange.message_id}")


def _answer(library: Library, conversations: ConversationStore,
            active: Optional[int], question: str) -> None:
    settings = library.info.chatbot
    if active is None:
        active = conversations.create()
        st.session_state.active_conversation = active

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the library and answering…"):
            try:
                chat_llm, ask_llm = state.make_llms(settings)
                result = ask(library, question, conversations.history(active),
                             chat_llm, ask_llm)
            except Exception as e:  # noqa: BLE001 -- UI boundary
                st.error(f"Could not get an answer from Ollama: {e}")
                return

    exchange = conversations.append(
        active, question, result.rag.answer, settings.chat_model,
        saved_sources(library, result.rag.sources), result.rag.citations,
        result.followups,
    )
    st.session_state.setdefault("diagnostics", {})[exchange.message_id] = result.rag.diagnostics
    st.rerun()
