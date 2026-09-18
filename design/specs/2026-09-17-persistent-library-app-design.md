# Persistent Library App — Design

**Date:** 2026-09-17
**Status:** Approved in brainstorming; awaiting implementation plan
**Scope:** A second Streamlit app, `library_app.py`, that keeps named document
libraries on disk and serves each one as a configurable, persistent chatbot.

---

## 1. Goals and non-goals

### Goals
- Build a document library once and query it across sessions without
  re-uploading or re-embedding.
- Detect duplicate uploads, and record publication details (title, authors,
  year, journal, DOI) so the library's contents can be browsed.
- Save a chatbot configuration (model, voice, instructions, retrieval settings)
  with each library.
- Save conversations with each library so they can be browsed and resumed.
- Reuse the existing retrieval engine (`rag_core`), including hybrid
  semantic + keyword search.

### Non-goals (this version)
- **Changing the existing app.** `app.py` stays byte-for-byte unchanged and
  keeps its "nothing is written to disk" guarantee.
- Multiple users, logins, hosting, or concurrent access by several people.
  These are personal tools: each person runs their own copy on their own
  hardware.
- Answer evaluation (LLM-as-judge). Evaluation is a separate activity.
- Question templates.
- Online metadata lookup (CrossRef or similar). The app stays fully offline.
- Keeping copies of original files.
- OCR for scanned documents.
- Searching, tagging, or foldering conversations.

### Target scale
"Medium" libraries: roughly 50–500 documents. A typical 15-page paper yields
about 50–100 chunks, so a library is expected to hold about 25,000–50,000
chunks. At this size the keyword (BM25) index stays in memory and is rebuilt
when a library opens (a few seconds). Moving it to SQLite FTS5 is a known
upgrade path for larger libraries and is not part of this version.

---

## 2. Architecture (Approach A)

A new app plus new, UI-free modules in `rag_core`. The existing app's code path
is not touched; the answer-orchestration logic that lives inside `app.py`
today is re-implemented once, properly, as a tested module that only the new
app uses for now.

```
rag_core/
  store.py             UNCHANGED  EphemeralRagStore (used by app.py)
  persistent_store.py  NEW  PersistentRagStore: Chroma PersistentClient +
                            chunks in SQLite + BM25 rebuilt on open
  library.py           NEW  create / open / list / rename / delete libraries;
                            add / remove / replace documents; duplicate checks;
                            publication-detail extraction; crash recovery
  conversations.py     NEW  save / list / resume / rename / delete conversations
  chat_session.py      NEW  ingest + ask + follow-up orchestration, no Streamlit
library_app.py         NEW  Streamlit UI
app.py                 UNCHANGED
```

Rules:
- New modules have no Streamlit imports, so they are testable without a
  browser, matching the existing `rag_core` design.
- Existing `rag_core` modules are reused as they are (chunker, embeddings,
  enrichment, query transformer, reranker, context assembler, citations,
  pipeline). If one needs a small, backward-compatible change, the existing
  test suite must keep passing unchanged.
- `PersistentRagStore` exposes the same search interface as
  `EphemeralRagStore`, so `rag_core.pipeline` works with either.
- `app.py` may migrate onto `chat_session.py` later, as a separate, deliberate
  change. Not part of this work.

---

## 3. Libraries and storage

### Location
- Default: `libraries/` in the app folder. Overridable with the environment
  variable `RAG_LIBRARY_DIR`.
- `libraries/` is added to `.gitignore` so library contents can never be
  committed.

### Folder layout (one folder per library)
```
libraries/<library-slug>/
  library.json   metadata + settings
  library.db     SQLite: documents, parents, chunks, conversations, messages
  chroma/        Chroma PersistentClient files
  .lock          present while a copy of the app has the library open
```
Original files are **not** kept.

### `library.json`
- `schema_version` (integer, starts at 1; future layout changes migrate from it)
- `name`, `description`, `created_at`
- `embedding_model` (fixed at creation), `embedding_dimensions`
- `chunking`: child chunk chars, overlap, parent max chars
- `contextual_headers` (bool, fixed at creation)
- `chatbot`: the settings in section 5

### `library.db` tables
- **documents**: `doc_id`, `filename`, `file_size`, `doc_type`,
  `page_or_slide_count`, `file_sha256`, `text_sha256`, `collection`,
  `added_at`, `chunk_count`, `complete` (bool), and publication details
  (`title`, `authors`, `year`, `journal`, `doi`), each with a `*_source` value
  of `doi_text`, `file_properties`, `llm`, `user`, or `none`.
- **parents**: `parent_id`, `doc_id`, `text`. Each parent passage is stored
  once, not repeated per child chunk.
- **chunks**: `chunk_id`, `doc_id`, `parent_id`, `text`, `chunk_index`,
  `char_start`, `char_end`, `page`, `slide`, `section`, `extra` (JSON,
  including any contextual header).
- **conversations**, **messages**: see section 6.

SQLite runs in WAL mode.

### Embedding model lock
The embedding model is chosen at creation and recorded. Opening a library whose
model is not installed shows the exact `ollama pull <model>` command instead of
searching with a mismatched model. Switching models is done only through
**Re-embed library** (section 4).

### Opening a library
1. Take the lock (section 8).
2. Run crash recovery: delete every document with `complete = false` from
   Chroma and SQLite, and report their filenames to the user.
3. Open the Chroma collection, load chunks from SQLite, rebuild BM25.
4. Keep the opened store cached for the app process (Streamlit
   `cache_resource`, keyed by library path).

### Write ordering
For each document: insert the document row with `complete = false` → write
parents and chunks to SQLite → add embeddings to Chroma → set `complete = true`.
A document is only visible to search once complete.

---

## 4. Adding and removing documents

### Two-step add
**Step 1: Check** (fast). The user selects files and a collection (existing or
new name). For each file:
1. Extract text with the existing `DocumentProcessor`.
2. Compute `file_sha256` and `text_sha256`. The text hash is taken over the
   extracted text lowercased with whitespace and punctuation removed.
3. Read file properties (PDF info/XMP, DOCX/PPTX core properties).
4. Search the first pages for a DOI (regex).
5. Compare against the library.

Results appear in a review table:

| Match | Status | Default action |
|---|---|---|
| Same `file_sha256` | Exact duplicate of *<existing title>* | Skipped (automatic) |
| Same `text_sha256` | Same content as *<existing title>* | Skipped (automatic) |
| Same DOI | Possible duplicate | User chooses: Skip / Replace / Keep both |
| Same normalized title + year (no DOI) | Possible duplicate | User chooses: Skip / Replace / Keep both |
| No text extracted | "No text found (scanned?)" | Not added |
| Otherwise | New | Add |

**Step 2: Add** (slow). For each accepted file, in order:
1. LLM extracts publication details from the first page (one call).
2. Chunk with the library's chunk settings.
3. If the library has contextual headers on, run the existing enrichment.
4. Embed with the library's embedding model.
5. Save per the write ordering in section 3.

Progress shows the file number and embedding progress. Each file is committed
independently, so an interruption loses only the file in progress. Closing the
browser tab stops the batch (a Streamlit limitation); crash recovery cleans up.

### Publication details: precedence
- `user` edits always win.
- DOI: `doi_text` > `llm` > `file_properties`.
- Title, authors, year, journal: `llm` > `file_properties`.
- If the LLM call fails or returns nothing usable, fields fall back to file
  properties or stay blank, and the document is still added.
- Manual editing is available via an "Edit details" action on a document; it
  is not a prominent part of the screen.

### Contextual headers
Chosen once, at library creation. Off by default. The creation form shows the
cost: one model call per chunk (roughly 75 calls for a 15-page paper). All
chunks in a library are treated the same way.

### Collections
Each document belongs to exactly one collection. Documents can be moved between
collections; collections can be renamed. Collection is stored as chunk metadata
in Chroma so it can be used as a pre-search filter; moving a document or
renaming a collection updates that metadata on the affected chunks (no
re-embedding).

### Remove, replace, re-embed
- **Remove:** delete the document's chunks from Chroma and SQLite (parents and
  chunks), delete the document row, rebuild BM25. Conversations are unaffected
  (they keep passage copies, section 6).
- **Replace:** remove, then add.
- **Re-embed library:** choose a new embedding model; re-embed every stored
  chunk into a new Chroma collection, then swap it in and update
  `library.json`. No re-uploading needed. If interrupted, the old collection
  stays in use.

---

## 5. Chatbot settings

Saved in `library.json` under `chatbot`, edited in the **Chatbot settings**
tab, and saved immediately.

| Group | Settings | Default |
|---|---|---|
| Identity | Name, description, custom instructions | Library name; empty; empty |
| Voice | Perspective, audience | Same as `app.py` defaults |
| Model | Chat model, temperature, top-K, top-P, context length | Same ranges/defaults as `app.py` |
| Answering | Allow LLM general knowledge | Off |
| | Generate follow-up questions | On |
| Retrieval | Semantic/keyword balance, passages sent, candidates considered, candidates per index | Same as `app.py` |
| | Resolve follow-up references, rewrite query, multiple phrasings, HyDE, infer filter | Same as `app.py` |
| | Re-rank retrieved passages | **On** (LLM re-ranker) |
| | Collections searched by default | All |

Rules:
- **Custom instructions are appended** to the built-in system prompt. They
  cannot replace or disable the grounding rules or the `[S1]` citation rules.
- No per-conversation overrides: a library always answers with its saved
  settings.
- The chat screen shows a one-line summary: chatbot name · chat model ·
  perspective → audience · collections.
- If the saved chat model is not installed, the user is asked to pick another,
  with an option to save it.

---

## 6. Conversations

### Storage (`library.db`)
- **conversations**: `conversation_id`, `title`, `created_at`, `updated_at`.
- **messages** (one row per exchange): `message_id`, `conversation_id`,
  `position`, `question`, `answer`, `chat_model`, `created_at`, `sources`
  (JSON list of `{marker, doc_id, location_label, passage}` where `passage` is
  a copy of up to 600 characters), `citation_check` (JSON result, including
  any fabricated markers), `followups` (JSON list).

### Behavior
- Each exchange is saved as soon as its answer completes.
- **New chat** starts an empty conversation using the current settings.
- **Title** defaults to the first question, shortened (no model call); can be
  renamed.
- **Resume** reloads the conversation; the last 5 exchanges are passed to the
  model as history, as in `app.py`.
- Follow-up suggestions stay clickable after resuming.
- **Delete** requires confirmation.
- **Export to HTML**: whole conversation or single Q&A, using
  `utils/html_export.py` (no evaluation cards).
- Retrieval Diagnostics are shown for answers in the current session only and
  are not saved.
- Cited passages remain in conversation history after their document is
  removed; deleting the conversation or library removes them.

---

## 7. Screens

Sidebar:
- Library picker, and **+ New library**.
- Tabs: 💬 Chat · 📄 Documents · ⚙️ Chatbot settings · 🗄️ Library settings.
- On the Chat tab: conversation list (newest first) with **+ New chat**.

Main area:
- **No libraries yet:** Create library form (name, embedding model, contextual
  headers with cost note).
- **Chat:** settings summary line, conversation with sources and follow-ups,
  question input.
- **Documents:** the two-step add flow; a table of title, authors, year,
  journal, collection, chunks, date added; filter by collection; Edit details;
  Remove; Move to collection.
- **Chatbot settings:** section 5.
- **Library settings:** rename; re-embed with another model; storage path and
  disk usage; delete library (requires typing the library's name).

---

## 8. Error handling

| Situation | Behavior |
|---|---|
| Ollama not running | Banner with instructions; browsing works, adding and asking are disabled |
| Embedding model missing | Show `ollama pull <model>`; library cannot be searched until installed |
| Chat model missing | Ask the user to pick another; optionally save it |
| File has no extractable text | "No text found (scanned?)" in review table; not added |
| LLM detail extraction fails | Fields fall back or stay blank; document still added |
| Crash / closed tab during add | Incomplete documents removed on next open and listed |
| Library already open in another app process | `.lock` file (holding the process ID) blocks it with a message; a lock whose process no longer exists is treated as stale and replaced. Multiple tabs in one process are fine |
| Re-embed interrupted | Previous collection remains active |

---

## 9. Testing

Unit tests need no Ollama; they use a stand-in embedder and a stubbed LLM, as
the existing tests do.

- Library lifecycle: create, list, open, rename, delete.
- **Persistence round trip:** add documents, close, reopen; the same query
  returns the same results from both the semantic and keyword indexes.
- Duplicate detection: each row of the section 4 table.
- DOI extraction from text; publication-detail precedence rules.
- Crash recovery: a document left `complete = false` is removed on open.
- Remove document: gone from both indexes; conversations keep passage copies.
- Re-embed: all chunks re-embedded; interruption leaves the old collection.
- Conversations: save, list, resume (history window of 5), rename, delete.
- Custom instructions are appended to, and do not replace, the built-in rules.
- Lock file: second process is refused; stale lock is recovered.
- **All existing tests (115) pass unchanged.**
- Manual: run `library_app.py` against real Ollama with a handful of PDFs.

---

## 10. Documentation

- README: a section on the library app — what it is, how to run
  (`streamlit run library_app.py`), where libraries are stored, and how to back
  one up (copy its folder).
- `docs/`: a new plain-language page on libraries, linked from the site's
  navigation.

---

## 11. Decisions log

| Decision | Choice |
|---|---|
| Separate app vs. mode in current app | Separate app, same repo, shared `rag_core` |
| Shared orchestration | New `chat_session.py`; `app.py` untouched (Approach A) |
| Audience | Personal, single user, local hardware |
| Scale | Medium (50–500 documents) |
| Saved with a library | Documents + index, chatbot settings, conversations |
| Evaluation | Not included |
| Original files | Not kept; metadata and extracted text only |
| Metadata sources | DOI from text, file properties, LLM (on); CrossRef (not built); manual edit (available, low prominence) |
| Contextual headers | Per library, chosen at creation, off by default |
| Re-ranking default | On |
| Per-conversation setting overrides | None |
| Cited passages in history | Copied (up to 600 chars) |
| Question templates | Not included |
| Spec location | `design/specs/` (keeps specs off the docs website) |
