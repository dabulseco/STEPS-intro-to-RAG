"""Documents tab: check → review → add, the document table, and edits."""

from typing import Callable, List, NamedTuple, Optional

import streamlit as st

from rag_core.library import CheckResult, Library
from utils.document_processor import DocumentProcessor

from . import settings_tabs, state

_NEW_COLLECTION = "➕ New collection…"
_STATUS_ICON = {
    "new": "✅", "duplicate": "⏭️", "possible_duplicate": "⚠️",
    "no_text": "🚫", "error": "❌",
}
_ACTIONS = {"skip": "Skip", "replace": "Replace existing", "keep_both": "Keep both"}


class Blocked(NamedTuple):
    """Why documents cannot be added right now."""

    kind: str       # 'ollama' | 'embedding' | 'chat_model'
    message: str


def render(library: Library, status: state.OllamaStatus, blocked: Optional[Blocked],
           ask_llm: Optional[Callable[[str], str]]) -> None:
    st.header("📄 Documents")
    _flash()
    _render_add(library, status, blocked, ask_llm)
    st.markdown("---")
    _render_table(library)


def _flash() -> None:
    message = st.session_state.pop("documents_flash", None)
    if message:
        kind, text = message
        getattr(st, kind)(text)


# -----------------------------------------------------------------------------
# Adding
# -----------------------------------------------------------------------------
def _render_add(library: Library, status: state.OllamaStatus,
                blocked: Optional[Blocked], ask_llm) -> None:
    st.subheader("Add documents")
    if blocked is not None:
        st.warning(blocked.message)
        if blocked.kind == "chat_model":
            settings_tabs.render_chat_model_picker(library, status, key="docs_chat_model")
        return
    if ask_llm is None:
        st.caption("No chat model is set, so titles and authors come only from the "
                   "files' stored properties. Choose a chat model in Chatbot settings "
                   "for better details.")

    pending: Optional[List[CheckResult]] = st.session_state.get("pending_checks")
    if pending is None:
        _render_upload(library)
    else:
        _render_review(library, pending, ask_llm)


def _render_upload(library: Library) -> None:
    uploader_key = f"uploader_{st.session_state.get('uploader_nonce', 0)}"
    files = st.file_uploader(
        "PDF, DOCX or PPTX", type=["pdf", "docx", "pptx"],
        accept_multiple_files=True, key=uploader_key,
    )
    existing = library.collections()
    options = existing + [_NEW_COLLECTION] if existing else [_NEW_COLLECTION]
    choice = st.selectbox("Collection", options)
    collection = choice
    if choice == _NEW_COLLECTION:
        collection = st.text_input("New collection name", value="" if existing else "General")

    if library.info.contextual_headers:
        st.caption("This library adds contextual headers: about one model call per "
                   "chunk, so adding takes considerably longer.")

    if st.button("Check files", type="primary", disabled=not files):
        if not collection.strip():
            st.error("Name the collection.")
            return
        items = []
        with st.spinner("Reading files and checking for duplicates…"):
            for f in files:
                data = f.getvalue()
                items.append((f.name, data, DocumentProcessor.process_document(f)))
            st.session_state.pending_checks = library.check_many(items)
            st.session_state.pending_collection = collection.strip()
        st.rerun()


def _render_review(library: Library, pending: List[CheckResult], ask_llm) -> None:
    collection = st.session_state.get("pending_collection", "General")
    st.write(f"Review before adding to **{collection}**:")

    for i, check in enumerate(pending):
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"{_STATUS_ICON.get(check.status, '')} **{check.filename}**  \n"
                    f"<small>{check.message}</small>", unsafe_allow_html=True)
        if check.status == "possible_duplicate":
            choice = c2.selectbox(
                "Action", list(_ACTIONS), format_func=_ACTIONS.get,
                key=f"dup_action_{i}", label_visibility="collapsed",
            )
            check.action = choice
        elif check.status == "new":
            c2.markdown("Add")
        else:
            c2.markdown("Skipped")

    addable = [c for c in pending if c.can_add]
    b1, b2 = st.columns([1, 1])
    cancel = b2.button("Cancel")
    go = b1.button(f"Add {len(addable)} document{'s' if len(addable) != 1 else ''}",
                   type="primary", disabled=not addable)
    if cancel:
        _clear_pending()
        st.rerun()
    if go:
        _run_add(library, addable, collection, ask_llm)


def _run_add(library: Library, checks: List[CheckResult], collection: str, ask_llm) -> None:
    added, failed = [], []
    overall = st.progress(0.0)
    stage_bar = st.progress(0.0)
    labels = {"details": "Reading details", "context": "Adding context",
              "embedding": "Embedding"}
    for n, check in enumerate(checks, start=1):
        overall.progress((n - 1) / len(checks),
                         text=f"File {n} of {len(checks)}: {check.filename}")

        def report(stage, done, total, _name=check.filename):
            stage_bar.progress(min(done / total, 1.0) if total else 1.0,
                               text=f"{labels.get(stage, stage)}… {done}/{total}")
        try:
            record = library.add(check, collection, ask_llm=ask_llm, progress=report)
            added.append(record.display_title)
        except Exception as e:  # noqa: BLE001 -- per-file failure is reported, batch continues
            failed.append(f"{check.filename}: {e}")
    overall.empty()
    stage_bar.empty()

    parts = []
    if added:
        parts.append(f"Added {len(added)} document{'s' if len(added) != 1 else ''}.")
    if failed:
        parts.append("Failed: " + "; ".join(failed))
    st.session_state.documents_flash = ("error" if failed and not added else
                                        "warning" if failed else "success", " ".join(parts))
    _clear_pending()
    st.rerun()


def _clear_pending() -> None:
    st.session_state.pop("pending_checks", None)
    st.session_state.pop("pending_collection", None)
    st.session_state.uploader_nonce = st.session_state.get("uploader_nonce", 0) + 1


# -----------------------------------------------------------------------------
# Table and edits
# -----------------------------------------------------------------------------
def _render_table(library: Library) -> None:
    documents = library.documents()
    st.subheader(f"In this library ({len(documents)})")
    if not documents:
        st.info("No documents yet.")
        return

    collections = library.collections()
    shown = documents
    if len(collections) > 1:
        pick = st.selectbox("Show collection", ["All"] + collections)
        if pick != "All":
            shown = [d for d in documents if d.collection == pick]

    st.dataframe(
        [{
            "Title": d.display_title,
            "Authors": d.authors,
            "Year": d.year,
            "Journal": d.journal,
            "Collection": d.collection,
            "Chunks": d.chunk_count,
            "Added": d.added_at[:10],
            "File": d.filename,
        } for d in shown],
        use_container_width=True, hide_index=True,
    )

    with st.expander("Manage a document"):
        doc = st.selectbox("Document", shown, format_func=lambda d: f"{d.display_title} ({d.filename})")
        if doc is None:
            return

        st.caption("Details sources — " + ", ".join(
            f"{f}: {getattr(doc, f + '_source')}" for f in ("title", "authors", "year", "journal", "doi")
        ))
        with st.form(f"edit_{doc.doc_id}"):
            st.markdown("**Edit details**")
            title = st.text_input("Title", value=doc.title)
            authors = st.text_input("Authors", value=doc.authors)
            c1, c2 = st.columns(2)
            year = c1.text_input("Year", value=doc.year)
            doi = c2.text_input("DOI", value=doc.doi)
            journal = st.text_input("Journal", value=doc.journal)
            if st.form_submit_button("Save details"):
                edits = {k: v for k, v in dict(title=title, authors=authors, year=year,
                                               journal=journal, doi=doi).items()
                         if v.strip() != getattr(doc, k)}
                if edits:
                    library.edit_details(doc.doc_id, **edits)
                    st.session_state.documents_flash = ("success", "Details saved.")
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            targets = [c for c in collections if c != doc.collection] + [_NEW_COLLECTION]
            target = st.selectbox("Move to collection", targets, key=f"move_{doc.doc_id}")
            if target == _NEW_COLLECTION:
                target = st.text_input("New collection name", key=f"newcol_{doc.doc_id}")
            if st.button("Move", disabled=not target.strip()):
                library.move(doc.doc_id, target.strip())
                st.session_state.documents_flash = ("success", f"Moved to {target.strip()}.")
                st.rerun()
        with c2:
            sure = st.checkbox("I want to remove this document", key=f"rm_ok_{doc.doc_id}")
            if st.button("Remove from library", disabled=not sure):
                library.remove(doc.doc_id)
                st.session_state.documents_flash = (
                    "success", f"Removed {doc.display_title}. Past conversations keep "
                               "the passages they cited.")
                st.rerun()

    if len(collections) > 0:
        with st.expander("Rename a collection"):
            with st.form("rename_collection"):
                old = st.selectbox("Collection", collections)
                new = st.text_input("New name")
                if st.form_submit_button("Rename") and new.strip() and new.strip() != old:
                    library.rename_collection(old, new.strip())
                    st.session_state.documents_flash = ("success", f"Renamed to {new.strip()}.")
                    st.rerun()
