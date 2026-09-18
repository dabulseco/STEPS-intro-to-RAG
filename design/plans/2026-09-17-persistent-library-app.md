# Persistent Library App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `library_app.py`, a second Streamlit app that keeps named document libraries on disk and serves each as a configurable chatbot with saved conversations.

**Architecture:** New UI-free modules in `rag_core/` (SQLite layer, persistent store, metadata extraction, library lifecycle, conversations, chat orchestration) plus a Streamlit front end split into `library_app.py` and a small `library_ui/` package. `PersistentRagStore` subclasses `EphemeralRagStore`, overriding only client/collection creation, so `rag_core.pipeline` works unchanged. `app.py` and `rag_core/store.py` are not modified.

**Tech Stack:** Python 3.11, Streamlit 1.62, chromadb 1.5 (`PersistentClient`), rank-bm25, SQLite (stdlib `sqlite3`, WAL), PyPDF2 / python-docx / python-pptx, langchain-ollama (UI layer only), pytest.

**Spec:** `design/specs/2026-09-17-persistent-library-app-design.md`

## Global Constraints

- `app.py` and `rag_core/store.py` must not change (verify with `git diff --stat` at the end).
- All 115 existing tests must pass unchanged after every task.
- New `rag_core` modules import neither Streamlit nor LangChain. LLM access is passed in as callables:
  - `ask_llm: Callable[[str], str]` — single prompt → text
  - `chat_llm: Callable[[List[Tuple[str, str]]], str]` — list of `(role, content)` with role in `{"system","user","assistant"}` → text
- Only `rag_core/chat_session.py` may import from `utils/` (for `PromptBuilder`); no other `rag_core` module imports `utils`.
- No network access except the local Ollama server. No CrossRef.
- Original files are never written to disk by the library.
- Library root: `libraries/` in the app folder, overridable by env var `RAG_LIBRARY_DIR`; `libraries/` is git-ignored.
- `library.json` carries `schema_version: 1`.
- Chroma collection names must be 3–512 chars of `[a-zA-Z0-9._-]`, starting and ending alphanumeric. Use `chunks_v1` (re-embed creates `chunks_v2`, …).
- Chroma collections are created with `embedding_function=None` (never download Chroma's default ONNX model).
- Conversation history window: last 5 exchanges. Cited passage copies: up to 600 characters.
- Tests need no Ollama: reuse the `StubEmbedder` pattern from `tests/test_rag_core.py` and stub LLM callables.

## File Structure

| File | Responsibility |
|---|---|
| `rag_core/library_db.py` | SQLite connection, schema v1, all SQL for documents / parents / chunks |
| `rag_core/persistent_store.py` | `PersistentRagStore`: Chroma on disk, load chunks without re-embedding, remove chunks, update chunk metadata |
| `rag_core/doc_metadata.py` | Fingerprints, DOI detection, file-property reading, LLM detail extraction, precedence merge, title normalization |
| `rag_core/library_lock.py` | PID lock file with stale-lock recovery |
| `rag_core/library.py` | Settings dataclasses, `library.json`, list/create/open/rename/delete, check → add pipeline, remove/replace/move, re-embed, crash recovery |
| `rag_core/conversations.py` | Conversations and messages (SQL + dataclasses) |
| `rag_core/chat_session.py` | System prompt (with custom instructions), `ask()` orchestration, follow-ups |
| `library_ui/__init__.py`, `library_ui/state.py` | Session-state helpers, cached library opening, LLM wiring |
| `library_ui/chat_tab.py` | Chat tab + conversation list |
| `library_ui/documents_tab.py` | Two-step add, document table, edit/move/remove |
| `library_ui/settings_tabs.py` | Chatbot settings, library settings, create-library form |
| `library_app.py` | Entry point: page config, sidebar, tab routing |
| `tests/library_fakes.py` | `StubEmbedder`, `FakeUpload`, stub LLMs shared by new tests |
| `tests/test_library_db.py`, `test_persistent_store.py`, `test_doc_metadata.py`, `test_library.py`, `test_conversations.py`, `test_chat_session.py`, `test_library_app.py` | Tests |
| `.gitignore`, `README.md`, `docs/libraries.html`, docs nav | Ignore rule and documentation |

---

### Task 1: SQLite layer (`rag_core/library_db.py`)

**Files:** Create `rag_core/library_db.py`, `tests/library_fakes.py`, `tests/test_library_db.py`

**Interfaces — Produces:**
```python
SCHEMA_VERSION = 1
@dataclass
class DocumentRecord:
    doc_id: str; filename: str; file_size: int; doc_type: str
    page_or_slide_count: int; file_sha256: str; text_sha256: str
    collection: str; added_at: str; chunk_count: int; complete: bool
    title: str = ""; title_source: str = "none"
    authors: str = ""; authors_source: str = "none"
    year: str = ""; year_source: str = "none"
    journal: str = ""; journal_source: str = "none"
    doi: str = ""; doi_source: str = "none"

class LibraryDB:
    def __init__(self, path: Path)            # connect, WAL, foreign_keys=ON, create schema
    def close(self) -> None
    def insert_document(self, rec: DocumentRecord) -> None
    def mark_complete(self, doc_id: str, chunk_count: int) -> None
    def get_document(self, doc_id: str) -> Optional[DocumentRecord]
    def list_documents(self, collection: Optional[str] = None, include_incomplete: bool = False) -> List[DocumentRecord]
    def incomplete_documents(self) -> List[DocumentRecord]
    def find_by_file_hash(self, sha: str) -> Optional[DocumentRecord]      # complete only
    def find_by_text_hash(self, sha: str) -> Optional[DocumentRecord]
    def find_by_doi(self, doi: str) -> Optional[DocumentRecord]             # case-insensitive
    def find_by_title_year(self, norm_title: str, year: str) -> Optional[DocumentRecord]
    def update_details(self, doc_id: str, **fields) -> None                # title/authors/year/journal/doi (+ *_source)
    def set_collection(self, doc_id: str, collection: str) -> None
    def rename_collection(self, old: str, new: str) -> List[str]           # returns affected doc_ids
    def collections(self) -> List[str]
    def insert_chunks(self, doc_id: str, chunks: Sequence[Chunk]) -> None  # parents deduped by parent_id
    def load_chunks(self, doc_id: Optional[str] = None) -> List[Chunk]     # rebuilds Chunk incl. parent_text, extra
    def chunk_ids(self, doc_id: str) -> List[str]
    def delete_document(self, doc_id: str) -> None                         # cascades parents + chunks
    connection: sqlite3.Connection                                          # shared with conversations.py
```
Schema (v1): `meta(key PRIMARY KEY, value)`; `documents(... columns above ..., norm_title TEXT)`; `parents(parent_id PK, doc_id FK ON DELETE CASCADE, text)`; `chunks(chunk_id PK, doc_id FK ON DELETE CASCADE, parent_id, text, chunk_index, char_start, char_end, page, slide, section, extra_json)`. Indexes on `documents(file_sha256)`, `(text_sha256)`, `(lower(doi))`, `(norm_title, year)`, `chunks(doc_id)`. Conversation tables are created by `conversations.py` (Task 5) on the same connection.

- [ ] Step 1: Create `tests/library_fakes.py` with `StubEmbedder` (copy of the one in `tests/test_rag_core.py`), `FakeUpload(name, data)` exposing `.name`, `.size`, `.read()`, `.getvalue()`, and `make_blocks(n, prefix)` producing page blocks with unique rare tokens.
- [ ] Step 2: Write failing tests: schema created and WAL on; insert → get round-trip; `complete=False` hidden from `list_documents` and hash lookups; `insert_chunks` → `load_chunks` returns equal `Chunk`s (text, parent_text, page, section, extra) with parents stored once; `delete_document` cascades; `set_collection` / `rename_collection` / `collections`; `find_by_doi` case-insensitive; `update_details`.
- [ ] Step 3: Run `pytest tests/test_library_db.py -q` → fails (module missing).
- [ ] Step 4: Implement `rag_core/library_db.py`.
- [ ] Step 5: Run new tests + full suite → pass.
- [ ] Step 6: Commit `feat(library): SQLite layer for persistent libraries`.

### Task 2: Persistent store (`rag_core/persistent_store.py`)

**Interfaces — Consumes:** `EphemeralRagStore`, `Chunk`. **Produces:**
```python
class PersistentRagStore(EphemeralRagStore):
    def __init__(self, embedder, chroma_path: Path, collection_name: str = "chunks_v1")
    def load_chunks(self, chunks: Sequence[Chunk]) -> None        # into memory + BM25, no embedding
    def missing_vector_ids(self, ids: Sequence[str]) -> List[str] # ids absent from Chroma
    def remove_chunks(self, ids: Sequence[str]) -> None           # Chroma + memory, BM25 rebuilt
    def update_chunk_metadata(self, chunks: Sequence[Chunk]) -> None  # Chroma metadata replaced from chunk.to_metadata()
    def close(self) -> None                                       # releases memory; never deletes the collection
```
Implementation: set `self._chroma_path` / `self._fixed_name` before `super().__init__`; override `_new_client` → `chromadb.PersistentClient(path=str(chroma_path))` and `_new_collection` → `get_or_create_collection(name=self._fixed_name, embedding_function=None, metadata={"hnsw:space": distance_metric})`; after `super().__init__` set `self.collection_name = self._fixed_name`. Add a module function `release_chroma(path)` calling `chromadb.api.client.SharedSystemClient.clear_system_cache()` for deletion.

- [ ] Step 1: Failing tests: add chunks, close, reopen new instance on same path + `load_chunks(db chunks)` → `search()` returns same top ids for a rare-token query (dense and sparse both non-empty); `remove_chunks` removes from both indexes; `update_chunk_metadata` makes a `where={"collection": "B"}` search find the chunk; `missing_vector_ids`; `close()` leaves vectors on disk.
- [ ] Step 2: Run → fail. Step 3: Implement. Step 4: Run full suite → pass.
- [ ] Step 5: Commit `feat(library): persistent Chroma-backed store`.

### Task 3: Document metadata (`rag_core/doc_metadata.py`)

**Produces:**
```python
def file_sha256(data: bytes) -> str
def normalize_text_for_hash(text: str) -> str      # lowercase, drop whitespace + punctuation
def text_sha256(text: str) -> str
def normalize_title(title: str) -> str             # lowercase, alnum + single spaces
def find_doi(text: str, max_chars: int = 20000) -> Optional[str]   # regex 10.\d{4,9}/..., strip trailing .,;)]
def read_file_properties(data: bytes, ext: str) -> Dict[str, str]  # keys: title, authors, year, doi (any may be absent); junk titles dropped
def page_or_slide_count(data: bytes, ext: str, blocks) -> int
def extract_details_llm(first_page_text: str, ask_llm) -> Dict[str, str]   # JSON reply parsed with the eval.parse_metric_payload fallback chain; {} on failure
@dataclass
class Detail: value: str; source: str
def merge_details(doi_text: Optional[str], file_props: Dict[str,str], llm: Dict[str,str]) -> Dict[str, Detail]
    # keys title, authors, year, journal, doi; doi: doi_text > llm > file_properties;
    # others: llm > file_properties; missing -> Detail("", "none"); year must match \d{4}
```
Junk title rule: drop titles that are empty, end in a file extension, start with "Microsoft Word -", or equal "untitled" (case-insensitive).

- [ ] Step 1: Failing tests: DOI found in prose / with trailing punctuation / none; text hash equal across whitespace + punctuation differences; normalize_title; junk-title filtering; DOCX core properties read (build a docx in-memory with python-docx, set `core_properties.title/author/created`); LLM extraction parses fenced JSON and returns {} for garbage; merge precedence table.
- [ ] Step 2: Run → fail. Step 3: Implement. Step 4: Run → pass. Step 5: Commit `feat(library): document fingerprints and publication details`.

### Task 4: Lock + library lifecycle (`rag_core/library_lock.py`, `rag_core/library.py`)

**Produces:**
```python
# library_lock.py
class LibraryLocked(RuntimeError)
class LibraryLock:
    def __init__(self, path: Path)
    def acquire(self) -> None      # writes PID; raises LibraryLocked if held by a live other PID; replaces stale lock
    def release(self) -> None
# library.py
LIBRARY_ROOT_ENV = "RAG_LIBRARY_DIR"
def library_root() -> Path
@dataclass class ChatbotSettings:   # defaults per spec §5; rerank=True, rerank_backend="llm",
    name, description, custom_instructions, perspective="Scientist", audience="Scientist",
    chat_model="", temperature=0.3, top_k=40, top_p=0.9, num_ctx=8192,
    allow_llm_knowledge=False, generate_followups=True,
    dense_weight, final_k, fused_k, dense_k (=sparse_k), condense_history=True,
    rewrite_query=False, multi_query=False, hyde=False, infer_filter=False,
    rerank=True, rerank_backend="llm", collections: List[str] = []   # [] = all
    def retrieval_settings(self) -> RetrievalSettings
@dataclass class LibraryInfo: schema_version, name, description, created_at, embedding_model,
    embedding_dimensions, chunking: Dict[str,int], contextual_headers: bool,
    collection_name: str = "chunks_v1", chatbot: ChatbotSettings
def slugify(name: str) -> str
def list_libraries(root: Optional[Path] = None) -> List[Tuple[Path, LibraryInfo]]
def create_library(name: str, embedding_model: str, contextual_headers: bool = False, root=None) -> Path
def read_info(path: Path) -> LibraryInfo;  def write_info(path: Path, info: LibraryInfo) -> None   # atomic write
def delete_library(path: Path) -> None     # refuses if locked by another live process
@dataclass class CheckResult:
    filename: str; status: str   # "new" | "duplicate" | "possible_duplicate" | "no_text" | "error"
    message: str; existing: Optional[DocumentRecord]; action: str   # "add" | "skip" | "replace" | "keep_both"
    file_sha256: str; text_sha256: str; doi: Optional[str]; file_props: Dict[str,str]
    processed: Dict; size: int; page_or_slide_count: int
class Library:
    path: Path; info: LibraryInfo; db: LibraryDB; store: PersistentRagStore; recovered: List[str]
    @classmethod
    def open(cls, path: Path, embedder=None) -> "Library"  # lock, recovery, load chunks, BM25
    def close(self) -> None
    def save_info(self) -> None
    def check(self, filename: str, data: bytes, processed: Dict) -> CheckResult
    def add(self, check: CheckResult, collection: str, ask_llm=None, progress=None) -> DocumentRecord
    def remove(self, doc_id: str) -> None
    def move(self, doc_id: str, collection: str) -> None
    def rename_collection(self, old: str, new: str) -> None
    def rename(self, new_name: str) -> None                 # display name only; folder unchanged
    def reembed(self, new_model: str, embedder=None, progress=None) -> None
    def disk_usage(self) -> int
```
Rules: `open` raises `EmbeddingModelMissing(model)` only via the UI check (Library itself accepts an injected embedder for tests; default `OllamaEmbedder(info.embedding_model)`). Recovery: for each incomplete document, remove its chunk ids from Chroma and delete the row; filenames returned in `recovered`. `add` write order per spec §3; chunk ids made unique per document with `chunk_document(..., doc_key=doc_id)`; `chunk.extra["collection"]` and `chunk.extra["doc_id"]` set before embedding; contextual headers run when `info.contextual_headers` and `ask_llm` given; `replace` action removes `check.existing` first. `reembed` builds `chunks_v{n+1}`, re-embeds all stored chunks, then updates `info.collection_name`, `embedding_model`, `embedding_dimensions`, deletes the old collection; on exception the old collection stays.

- [ ] Step 1: Failing tests (`tests/test_library.py`, `tmp_path` root via env var): create/list/read_info round-trip and slug collisions (`biol-201`, `biol-201-2`); lock blocks a second `Library.open` in-process when PID differs (simulate by writing another live PID = parent PID) and stale PID lock is replaced; `check` statuses for each spec §4 row; `add` then reopen finds document and search works; crash recovery (insert incomplete doc + vectors, reopen → gone, name in `recovered`); remove; move changes Chroma filter results; rename_collection; replace; reembed switches collection name and search still works; delete_library removes folder.
- [ ] Step 2: Run → fail. Step 3: Implement lock, then library. Step 4: Run full suite → pass.
- [ ] Step 5: Add `libraries/` to `.gitignore`. Commit `feat(library): library lifecycle, duplicate checks, recovery, re-embed`.

### Task 5: Conversations (`rag_core/conversations.py`)

**Produces:**
```python
@dataclass class SavedSource: marker: str; doc_id: str; location_label: str; passage: str
@dataclass class Exchange: message_id: int; position: int; question: str; answer: str; chat_model: str;
    created_at: str; sources: List[SavedSource]; citation_summary: str; citation_clean: bool;
    hallucinated: List[int]; followups: List[str]
@dataclass class ConversationSummary: conversation_id: int; title: str; created_at: str; updated_at: str; exchanges: int
class ConversationStore:
    def __init__(self, connection: sqlite3.Connection)   # creates tables
    def create(self, title: str = "New chat") -> int
    def list(self) -> List[ConversationSummary]           # newest updated first
    def exchanges(self, conversation_id: int) -> List[Exchange]
    def append(self, conversation_id: int, question, answer, chat_model, sources: Sequence[SavedSource],
               citations: Optional[CitationReport], followups: Sequence[str]) -> Exchange
               # first exchange sets title = question shortened to 60 chars if title is "New chat"
    def rename(self, conversation_id: int, title: str) -> None
    def delete(self, conversation_id: int) -> None
    def history(self, conversation_id: int, window: int = 5) -> List[Tuple[str, str]]
```
`SavedSource` objects are built by the caller (Task 6, `saved_sources()`); the passage is truncated to 600 chars inside `append` as a safeguard.

- [ ] Step 1: Failing tests: create/list order; append stores sources (truncated to 600), citation info, followups; title auto-set; history window of 5; rename; delete cascades messages; passages survive `Library.remove` of the cited document.
- [ ] Step 2–4: fail → implement → pass. Step 5: Commit `feat(library): saved conversations`.

### Task 6: Chat orchestration (`rag_core/chat_session.py`)

**Produces:**
```python
def build_system_prompt(settings: ChatbotSettings, has_documents: bool, grounding_mode: str) -> str
    # PromptBuilder.build_system_prompt(...) + ("\n\nAdditional instructions from the user:\n" + custom) — appended, never replacing
def generate_followups(question: str, answer: str, ask_llm) -> List[str]   # same prompt/parsing as app.py
def saved_sources(library: Library, sources: Sequence[Source]) -> List[SavedSource]
    # doc_id from library.store.get_chunk(source.chunk_id).extra["doc_id"] ("" if the chunk is gone); passage = source.text[:600]
def collection_filter(settings: ChatbotSettings, all_collections: Sequence[str]) -> Optional[Dict]
@dataclass class AskResult: rag: RagResult; followups: List[str]
def ask(library: Library, question: str, history: Sequence[Tuple[str,str]],
        chat_llm, ask_llm) -> AskResult
    # mode via resolve_mode; transformer/reranker built like app.run_query; where from collection_filter;
    # generate() composes [("system", prompt)] + history[-5:] + [("user", message)] and calls chat_llm
```
- [ ] Step 1: Failing tests: custom instructions appended after grounding text and grounding text still present; empty instructions add nothing; `collection_filter` None for all / `$in` for subset; `ask` with stub LLMs returns an answer citing `[S1]` and sources from the library; history passed is last 5; follow-ups parsed; strict mode with empty library declines.
- [ ] Step 2–4: fail → implement → pass. Step 5: Commit `feat(library): chat orchestration with custom instructions`.

### Task 7: Streamlit UI (`library_app.py`, `library_ui/`)

**Consumes:** everything above; `DocumentProcessor`, `html_export`, `rag_core.ollama_api`.

Behaviour per spec §7–8:
- `state.py`: `get_library(path)` via `st.cache_resource` keyed by path (opens `Library`, surfaces `LibraryLocked`); `close_cached_library()`; `make_llms(settings) -> (chat_llm, ask_llm)` wrapping `ChatOllama`; `ollama_status()` → (chat_models, embed_models, error).
- Sidebar: library picker (`list_libraries`), "+ New library", tab radio (Chat / Documents / Chatbot settings / Library settings), conversation list on Chat.
- No libraries / "+ New library": create form (name, embedding model from installed list, contextual-headers checkbox with cost note).
- Chat: summary line, exchanges with sources expander, citation warning, clickable follow-ups, diagnostics only for current-session answers, chat input; HTML export (conversation + per Q&A).
- Documents: uploader + collection selector → "Check files" → review table with per-row action select for possible duplicates → "Add N documents" with progress; table (title, authors, year, journal, collection, chunks, added) filterable by collection; per-document Edit details / Move / Remove.
- Chatbot settings: all fields from `ChatbotSettings`, saved on "Save".
- Library settings: rename, re-embed (model select + button), storage path + disk usage, delete (type name to confirm).
- Errors: Ollama down banner; embedding model missing → `ollama pull` message, chat/add disabled; chat model missing → prompt to pick; recovery notice when `library.recovered`.

- [ ] Step 1: Failing smoke test `tests/test_library_app.py` using `streamlit.testing.v1.AppTest.from_file("library_app.py")` with `RAG_LIBRARY_DIR` = tmp and Ollama unreachable (`OLLAMA_HOST=http://127.0.0.1:9`): runs without exception and shows the create form; with a pre-created library (StubEmbedder not needed to render) the Documents tab renders.
- [ ] Step 2: Implement UI modules. Step 3: Run smoke test + full suite → pass.
- [ ] Step 4: Manual run against real Ollama: create library, add 2–3 PDFs, duplicate re-upload skipped, ask questions, restart app, resume conversation.
- [ ] Step 5: Commit `feat(library): Streamlit library app`.

### Task 8: Documentation and final verification

- [ ] Step 1: README section "📚 Persistent Library App" (what, run command, storage location + `RAG_LIBRARY_DIR`, backup = copy folder, privacy note that libraries hold extracted text); project-structure entries.
- [ ] Step 2: `docs/libraries.html` in the site's style, linked from every page's nav and the index.
- [ ] Step 3: `git diff --stat main@{start}` shows `app.py` and `rag_core/store.py` unchanged; full test suite passes.
- [ ] Step 4: Commit `docs(library): README and documentation page`.

---

## Self-review notes

- Spec coverage: §3 → Tasks 1, 2, 4; §4 → Tasks 3, 4, 7; §5 → Tasks 4, 6, 7; §6 → Task 5, 7; §7 → Task 7; §8 → Tasks 4, 7; §9 → tests in every task; §10 → Task 8.
- Deviation from spec file list: metadata extraction (`doc_metadata.py`), the lock (`library_lock.py`) and the SQL layer (`library_db.py`) are split out of `library.py`, and the UI is split into `library_ui/`, to keep files focused. Behaviour is unchanged.
- `Source` has no `doc_id`; `chat_session.saved_sources()` resolves it from the stored chunk's `extra["doc_id"]`, which `Library.add` sets on every chunk before embedding.
