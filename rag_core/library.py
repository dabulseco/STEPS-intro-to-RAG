"""
Persistent document libraries.

A library is a folder:

    libraries/<slug>/
      library.json   name, embedding model, chunk settings, chatbot settings
      library.db     documents, chunks, conversations (SQLite)
      chroma/        vectors
      .lock          present while a copy of the app has it open

Original files are never stored. Everything here is free of Streamlit and
LangChain; LLM access is passed in as `ask_llm(prompt) -> str`.
"""

import datetime
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .chunker import chunk_document
from .config import (
    CHILD_CHUNK_CHARS,
    CHILD_CHUNK_OVERLAP,
    DENSE_TOP_K,
    DENSE_WEIGHT,
    FINAL_TOP_K,
    FUSED_TOP_K,
    MIN_CHUNK_CHARS,
    PARENT_MAX_CHARS,
)
from .doc_metadata import (
    extract_details_llm,
    file_sha256,
    find_doi,
    first_page_text,
    merge_details,
    normalize_title,
    page_or_slide_count,
    read_file_properties,
    text_sha256,
)
from .embeddings import OllamaEmbedder
from .enrichment import add_contextual_headers, summarize_document
from .library_db import DETAIL_FIELDS, DocumentRecord, LibraryDB
from .library_lock import LibraryLock, LibraryLocked, held_by_other_process
from .persistent_store import PersistentRagStore, release_chroma
from .pipeline import RetrievalSettings

SCHEMA_VERSION = 1
LIBRARY_ROOT_ENV = "RAG_LIBRARY_DIR"
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "libraries"

# (stage, done, total) -- stage is 'details', 'context' or 'embedding'.
ProgressFn = Callable[[str, int, int], None]


def library_root() -> Path:
    """Where libraries live: $RAG_LIBRARY_DIR, else ./libraries next to the app."""
    env = os.environ.get(LIBRARY_ROOT_ENV)
    return Path(env).expanduser() if env else _DEFAULT_ROOT


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _from_dict(cls, data: Dict):
    """Build a dataclass from a dict, ignoring keys from newer versions."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in known})


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
@dataclass
class ChatbotSettings:
    """Everything that makes a library a particular chatbot (spec §5)."""

    name: str = ""
    description: str = ""
    custom_instructions: str = ""
    perspective: str = "Scientist"
    audience: str = "Scientist"
    chat_model: str = ""
    temperature: float = 0.3
    top_k: int = 40
    top_p: float = 0.9
    num_ctx: int = 8192
    allow_llm_knowledge: bool = False
    generate_followups: bool = True
    dense_weight: float = DENSE_WEIGHT
    final_k: int = FINAL_TOP_K
    fused_k: int = FUSED_TOP_K
    dense_k: int = DENSE_TOP_K          # also used for the keyword index
    condense_history: bool = True
    rewrite_query: bool = False
    multi_query: bool = False
    hyde: bool = False
    infer_filter: bool = False
    rerank: bool = True
    rerank_backend: str = "llm"
    collections: List[str] = field(default_factory=list)   # [] = all

    def retrieval_settings(self) -> RetrievalSettings:
        return RetrievalSettings(
            dense_k=self.dense_k,
            sparse_k=self.dense_k,
            fused_k=self.fused_k,
            final_k=self.final_k,
            dense_weight=self.dense_weight,
            condense_history=self.condense_history,
            rewrite_query=self.rewrite_query,
            multi_query=self.multi_query,
            hyde=self.hyde,
            infer_filter=self.infer_filter,
            rerank=self.rerank,
        )


@dataclass
class LibraryInfo:
    """Contents of library.json."""

    name: str
    embedding_model: str
    schema_version: int = SCHEMA_VERSION
    description: str = ""
    created_at: str = ""
    embedding_dimensions: Optional[int] = None
    chunking: Dict[str, int] = field(default_factory=lambda: {
        "child_chars": CHILD_CHUNK_CHARS,
        "child_overlap": CHILD_CHUNK_OVERLAP,
        "parent_chars": PARENT_MAX_CHARS,
        "min_chars": MIN_CHUNK_CHARS,
    })
    contextual_headers: bool = False
    collection_name: str = "chunks_v1"
    chatbot: ChatbotSettings = field(default_factory=ChatbotSettings)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "LibraryInfo":
        data = dict(data)
        chatbot = _from_dict(ChatbotSettings, data.pop("chatbot", {}))
        info = _from_dict(cls, data)
        info.chatbot = chatbot
        return info


def read_info(path: Path) -> LibraryInfo:
    return LibraryInfo.from_dict(json.loads((Path(path) / "library.json").read_text()))


def write_info(path: Path, info: LibraryInfo) -> None:
    """Atomic: write a temp file, then rename over the old one."""
    target = Path(path) / "library.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info.to_dict(), indent=2))
    os.replace(tmp, target)


# -----------------------------------------------------------------------------
# Library folders
# -----------------------------------------------------------------------------
def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "library"


def list_libraries(root: Optional[Path] = None) -> List[Tuple[Path, LibraryInfo]]:
    """Every readable library under the root, sorted by display name."""
    root = Path(root) if root else library_root()
    if not root.exists():
        return []
    found = []
    for folder in sorted(root.iterdir()):
        if (folder / "library.json").is_file():
            try:
                found.append((folder, read_info(folder)))
            except (OSError, ValueError, TypeError):
                continue
    return sorted(found, key=lambda pair: pair[1].name.lower())


def create_library(
    name: str,
    embedding_model: str,
    contextual_headers: bool = False,
    root: Optional[Path] = None,
) -> Path:
    """Create an empty library folder and return its path."""
    root = Path(root) if root else library_root()
    root.mkdir(parents=True, exist_ok=True)
    base = slugify(name)
    path, n = root / base, 2
    while path.exists():
        path, n = root / f"{base}-{n}", n + 1
    path.mkdir()
    info = LibraryInfo(
        name=name.strip(),
        embedding_model=embedding_model,
        created_at=_now(),
        contextual_headers=contextual_headers,
        chatbot=ChatbotSettings(name=name.strip()),
    )
    write_info(path, info)
    LibraryDB(path / "library.db").close()
    return path


def delete_library(path: Path) -> None:
    """Delete a library folder; refused while another app process has it open."""
    path = Path(path)
    if held_by_other_process(path / ".lock"):
        raise LibraryLocked("This library is open in another running copy of the app.")
    release_chroma()
    shutil.rmtree(path)


# -----------------------------------------------------------------------------
# Duplicate checking
# -----------------------------------------------------------------------------
@dataclass
class CheckResult:
    """The outcome of checking one file before it is added (spec §4 step 1)."""

    filename: str
    status: str            # new | duplicate | possible_duplicate | no_text | error
    message: str
    action: str            # add | skip | replace | keep_both
    processed: Dict = field(default_factory=dict)
    existing: Optional[DocumentRecord] = None
    file_sha256: str = ""
    text_sha256: str = ""
    doi: Optional[str] = None
    file_props: Dict[str, str] = field(default_factory=dict)
    size: int = 0
    page_or_slide_count: int = 0

    @property
    def can_add(self) -> bool:
        return self.action in ("add", "replace", "keep_both")


# -----------------------------------------------------------------------------
# An open library
# -----------------------------------------------------------------------------
class Library:
    """An opened library: settings, SQLite, and the hybrid store."""

    def __init__(self, path: Path, info: LibraryInfo, db: LibraryDB,
                 store: PersistentRagStore, lock: LibraryLock, recovered: List[str]):
        self.path = path
        self.info = info
        self.db = db
        self.store = store
        self.recovered = recovered
        self._lock = lock
        self._closed = False

    @classmethod
    def open(cls, path: Path, embedder=None) -> "Library":
        """Lock, recover from any interrupted add, load chunks, rebuild BM25."""
        path = Path(path)
        info = read_info(path)
        lock = LibraryLock(path / ".lock")
        lock.acquire()
        db = None
        try:
            db = LibraryDB(path / "library.db")
            store = PersistentRagStore(
                embedder or OllamaEmbedder(info.embedding_model),
                path / "chroma",
                collection_name=info.collection_name,
            )
            recovered = []
            for doc in db.incomplete_documents():
                store.remove_chunks(db.chunk_ids(doc.doc_id))
                db.delete_document(doc.doc_id)
                recovered.append(doc.filename)
            store.load_chunks(db.load_chunks())
        except Exception:
            if db is not None:
                db.close()
            lock.release()
            raise
        return cls(path, info, db, store, lock, recovered)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.store.close()
        self.db.close()
        self._lock.release()

    def save_info(self) -> None:
        write_info(self.path, self.info)

    def rename(self, new_name: str) -> None:
        """Change the display name. The folder keeps its original slug."""
        self.info.name = new_name.strip()
        self.save_info()

    # --- queries -------------------------------------------------------------
    def documents(self, collection: Optional[str] = None) -> List[DocumentRecord]:
        return self.db.list_documents(collection=collection)

    def collections(self) -> List[str]:
        return self.db.collections()

    def disk_usage(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())

    # --- checking --------------------------------------------------------------
    def check(self, filename: str, data: bytes, processed: Dict) -> CheckResult:
        """Classify one extracted file against the library (spec §4 table)."""
        return self.check_many([(filename, data, processed)])[0]

    def check_many(self, items: Sequence[Tuple[str, bytes, Dict]]) -> List[CheckResult]:
        """Check a batch; later copies of a file earlier in the batch are duplicates."""
        results: List[CheckResult] = []
        seen_file: Dict[str, str] = {}
        seen_text: Dict[str, str] = {}
        for filename, data, processed in items:
            result = self._check_one(filename, data, processed)
            if result.status == "new":
                earlier = seen_file.get(result.file_sha256) or seen_text.get(result.text_sha256)
                if earlier:
                    result.status, result.action = "duplicate", "skip"
                    result.message = f"Same file as {earlier}, earlier in this batch"
                else:
                    seen_file[result.file_sha256] = filename
                    seen_text[result.text_sha256] = filename
            results.append(result)
        return results

    def _check_one(self, filename: str, data: bytes, processed: Dict) -> CheckResult:
        content = processed.get("content") or ""
        base = dict(filename=filename, processed=processed, size=len(data))

        if processed.get("error") or not content.strip():
            error = processed.get("error") or ""
            if not content.strip() and ("No text" in error or not error):
                return CheckResult(status="no_text", action="skip",
                                   message="No text found (scanned?)", **base)
            return CheckResult(status="error", action="skip", message=error, **base)

        ext = processed.get("file_type") or os.path.splitext(filename)[1].lower()
        result = CheckResult(
            status="new", action="add", message="New",
            file_sha256=file_sha256(data),
            text_sha256=text_sha256(content),
            doi=find_doi(content),
            file_props=read_file_properties(data, ext),
            page_or_slide_count=page_or_slide_count(processed.get("blocks") or []),
            **base,
        )
        if processed.get("warning"):
            result.message = f"New — {processed['warning']}"

        def flag(status: str, existing: DocumentRecord, why: str):
            result.status, result.action, result.existing = status, "skip", existing
            result.message = f"{why} {existing.display_title} ({existing.filename})"
            return result

        existing = self.db.find_by_file_hash(result.file_sha256)
        if existing:
            return flag("duplicate", existing, "Exact duplicate of")
        existing = self.db.find_by_text_hash(result.text_sha256)
        if existing:
            return flag("duplicate", existing, "Same content as")
        doi = result.doi or result.file_props.get("doi")
        existing = self.db.find_by_doi(doi) if doi else None
        if existing:
            return flag("possible_duplicate", existing, "Same DOI as")
        props = result.file_props
        existing = self.db.find_by_title_year(
            normalize_title(props.get("title", "")), props.get("year", "")
        )
        if existing:
            return flag("possible_duplicate", existing, "Same title and year as")
        return result

    # --- adding ----------------------------------------------------------------
    def add(
        self,
        check: CheckResult,
        collection: str,
        ask_llm: Optional[Callable[[str], str]] = None,
        progress: Optional[ProgressFn] = None,
    ) -> DocumentRecord:
        """
        Add one checked file (spec §4 step 2).

        Write order: document row (incomplete) → chunks in SQLite → vectors in
        Chroma → mark complete. A failure removes everything written so far;
        a crash leaves an incomplete row that the next open() cleans up.
        """
        if not check.can_add:
            raise ValueError(f"{check.filename} is marked '{check.action}', not addable.")
        if check.action == "replace" and check.existing is not None:
            self.remove(check.existing.doc_id)

        processed = check.processed
        blocks = processed.get("blocks") or []
        collection = (collection or "General").strip() or "General"
        report = progress or (lambda stage, done, total: None)

        report("details", 0, 1)
        llm_details = extract_details_llm(first_page_text(blocks), ask_llm)
        details = merge_details(check.doi, check.file_props, llm_details)
        report("details", 1, 1)

        doc_id = uuid.uuid4().hex[:12]
        record = DocumentRecord(
            doc_id=doc_id,
            filename=check.filename,
            file_size=check.size,
            doc_type=processed.get("file_type") or "",
            page_or_slide_count=check.page_or_slide_count,
            file_sha256=check.file_sha256,
            text_sha256=check.text_sha256,
            collection=collection,
            added_at=_now(),
            chunk_count=0,
            complete=False,
            **{k: details[k].value for k in DETAIL_FIELDS},
            **{f"{k}_source": details[k].source for k in DETAIL_FIELDS},
        )

        chunking = self.info.chunking
        chunks = chunk_document(
            blocks,
            source_file=check.filename,
            doc_type=record.doc_type,
            child_chars=chunking.get("child_chars", CHILD_CHUNK_CHARS),
            child_overlap=chunking.get("child_overlap", CHILD_CHUNK_OVERLAP),
            parent_chars=chunking.get("parent_chars", PARENT_MAX_CHARS),
            min_chars=chunking.get("min_chars", MIN_CHUNK_CHARS),
            doc_key=doc_id,
        )
        if not chunks:
            raise ValueError(f"{check.filename} produced no searchable text.")
        for chunk in chunks:
            chunk.extra["collection"] = collection
            chunk.extra["doc_id"] = doc_id

        self.db.insert_document(record)
        try:
            if self.info.contextual_headers and ask_llm is not None:
                summary = summarize_document(blocks, ask_llm)
                add_contextual_headers(
                    chunks, ask_llm,
                    {check.filename: summary} if summary else None,
                    progress_callback=lambda d, t: report("context", d, t),
                )
            self.db.insert_chunks(doc_id, chunks)
            self.store.add_chunks(
                chunks, progress_callback=lambda d, t: report("embedding", d, t)
            )
            self.db.mark_complete(doc_id, len(chunks))
        except BaseException:
            self.store.remove_chunks([c.chunk_id for c in chunks])
            self.db.delete_document(doc_id)
            raise

        if self.info.embedding_dimensions is None and getattr(self.store.embedder, "dimension", None):
            self.info.embedding_dimensions = self.store.embedder.dimension
            self.save_info()
        return self.db.get_document(doc_id)

    # --- changing ----------------------------------------------------------------
    def remove(self, doc_id: str) -> None:
        self.store.remove_chunks(self.db.chunk_ids(doc_id))
        self.db.delete_document(doc_id)

    def _relabel(self, doc_ids: Sequence[str], collection: str) -> None:
        for doc_id in doc_ids:
            chunks = self.db.load_chunks(doc_id)
            for chunk in chunks:
                chunk.extra["collection"] = collection
            self.store.update_chunk_metadata(chunks)
            self.db.update_chunk_extra(chunks)

    def move(self, doc_id: str, collection: str) -> None:
        self.db.set_collection(doc_id, collection)
        self._relabel([doc_id], collection)

    def rename_collection(self, old: str, new: str) -> None:
        self._relabel(self.db.rename_collection(old, new), new)
        chosen = self.info.chatbot.collections
        if old in chosen:
            self.info.chatbot.collections = [new if c == old else c for c in chosen]
            self.save_info()

    def edit_details(self, doc_id: str, **values: str) -> None:
        """Manual edits; each edited field is marked as coming from the user."""
        updates = {}
        for key, value in values.items():
            if key not in DETAIL_FIELDS:
                raise ValueError(f"Not an editable detail: {key}")
            updates[key] = value.strip()
            updates[f"{key}_source"] = "user" if value.strip() else "none"
        self.db.update_details(doc_id, **updates)

    def reembed(self, new_model: str, embedder=None, progress: Optional[ProgressFn] = None) -> None:
        """
        Re-embed every stored chunk into a new collection, then swap it in.

        No re-upload is needed: chunk text lives in SQLite. If embedding
        fails, the new collection is dropped and the old one stays in use.
        """
        match = re.search(r"(\d+)$", self.info.collection_name)
        new_name = f"chunks_v{int(match.group(1)) + 1 if match else 2}"
        report = progress or (lambda stage, done, total: None)

        new_store = PersistentRagStore(
            embedder or OllamaEmbedder(new_model), self.path / "chroma",
            collection_name=new_name,
        )
        try:
            new_store.add_chunks(
                self.db.load_chunks(), progress_callback=lambda d, t: report("embedding", d, t)
            )
        except BaseException:
            new_store.drop_collection()
            raise

        old_store = self.store
        self.info.collection_name = new_name
        self.info.embedding_model = new_model
        self.info.embedding_dimensions = getattr(new_store.embedder, "dimension", None)
        self.save_info()
        self.store = new_store
        old_store.drop_collection()
