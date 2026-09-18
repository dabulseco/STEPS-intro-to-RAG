"""
SQLite storage for a persistent library.

One `library.db` per library holds everything except the vectors: the document
list with its publication details, and every chunk with its parent passage.
Chroma keeps the vectors; this file is the source of truth for everything the
keyword index and the UI need, which is why a library can be reopened without
re-embedding anything.

Conversation tables live in the same file but are owned by
`rag_core.conversations`, which receives this module's connection.
"""

import json
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional, Sequence

from .chunker import Chunk

SCHEMA_VERSION = 1

# Publication-detail columns that may be edited after a document is added.
DETAIL_FIELDS = ("title", "authors", "year", "journal", "doi")
_EDITABLE = set(DETAIL_FIELDS) | {f"{f}_source" for f in DETAIL_FIELDS}


def normalize_title(title: str) -> str:
    """Lowercase, letters and digits only, single spaces -- for duplicate matching."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())


@dataclass
class DocumentRecord:
    """One document in a library, as the documents table stores it."""

    doc_id: str
    filename: str
    file_size: int
    doc_type: str
    page_or_slide_count: int
    file_sha256: str
    text_sha256: str
    collection: str
    added_at: str
    chunk_count: int
    complete: bool
    title: str = ""
    title_source: str = "none"
    authors: str = ""
    authors_source: str = "none"
    year: str = ""
    year_source: str = "none"
    journal: str = ""
    journal_source: str = "none"
    doi: str = ""
    doi_source: str = "none"

    @property
    def display_title(self) -> str:
        """The title when known, otherwise the filename."""
        return self.title or self.filename


_DOC_COLUMNS = [f.name for f in fields(DocumentRecord)]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id              TEXT PRIMARY KEY,
    filename            TEXT NOT NULL,
    file_size           INTEGER NOT NULL,
    doc_type            TEXT NOT NULL,
    page_or_slide_count INTEGER NOT NULL,
    file_sha256         TEXT NOT NULL,
    text_sha256         TEXT NOT NULL,
    collection          TEXT NOT NULL,
    added_at            TEXT NOT NULL,
    chunk_count         INTEGER NOT NULL,
    complete            INTEGER NOT NULL,
    title               TEXT NOT NULL DEFAULT '',
    title_source        TEXT NOT NULL DEFAULT 'none',
    authors             TEXT NOT NULL DEFAULT '',
    authors_source      TEXT NOT NULL DEFAULT 'none',
    year                TEXT NOT NULL DEFAULT '',
    year_source         TEXT NOT NULL DEFAULT 'none',
    journal             TEXT NOT NULL DEFAULT '',
    journal_source      TEXT NOT NULL DEFAULT 'none',
    doi                 TEXT NOT NULL DEFAULT '',
    doi_source          TEXT NOT NULL DEFAULT 'none',
    norm_title          TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_documents_file_sha ON documents(file_sha256);
CREATE INDEX IF NOT EXISTS idx_documents_text_sha ON documents(text_sha256);
CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(lower(doi));
CREATE INDEX IF NOT EXISTS idx_documents_title_year ON documents(norm_title, year);

-- Parent passages are stored once; many child chunks point at each.
CREATE TABLE IF NOT EXISTS parents (
    parent_id TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    text      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    parent_id   TEXT NOT NULL,
    text        TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL,
    page        INTEGER,
    slide       INTEGER,
    section     TEXT,
    extra_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class LibraryDB:
    """
    Thin, synchronous access layer over one library's SQLite file.

    Streamlit runs each script rerun on a worker thread, so the connection is
    opened with check_same_thread=False and every operation is serialized by a
    lock. A personal, single-user app never contends for it meaningfully.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        with self._lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript(_SCHEMA)
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.connection.commit()

    @property
    def lock(self) -> threading.RLock:
        """Shared with ConversationStore so both serialize on one connection."""
        return self._lock

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    # -------------------------------------------------------------------------
    # Documents
    # -------------------------------------------------------------------------
    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        data = {name: row[name] for name in _DOC_COLUMNS}
        data["complete"] = bool(data["complete"])
        return DocumentRecord(**data)

    def insert_document(self, rec: DocumentRecord) -> None:
        values = asdict(rec)
        values["complete"] = int(rec.complete)
        values["norm_title"] = normalize_title(rec.title)
        columns = list(values.keys())
        with self._lock:
            self.connection.execute(
                f"INSERT INTO documents ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                [values[c] for c in columns],
            )
            self.connection.commit()

    def mark_complete(self, doc_id: str, chunk_count: int) -> None:
        with self._lock:
            self.connection.execute(
                "UPDATE documents SET complete = 1, chunk_count = ? WHERE doc_id = ?",
                (chunk_count, doc_id),
            )
            self.connection.commit()

    def _one(self, sql: str, params: Sequence) -> Optional[DocumentRecord]:
        with self._lock:
            row = self.connection.execute(sql, params).fetchone()
        return self._row_to_record(row) if row else None

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        return self._one("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))

    def list_documents(
        self, collection: Optional[str] = None, include_incomplete: bool = False
    ) -> List[DocumentRecord]:
        clauses, params = [], []
        if not include_incomplete:
            clauses.append("complete = 1")
        if collection is not None:
            clauses.append("collection = ?")
            params.append(collection)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self.connection.execute(
                f"SELECT * FROM documents {where} ORDER BY added_at, filename", params
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def incomplete_documents(self) -> List[DocumentRecord]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM documents WHERE complete = 0"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def find_by_file_hash(self, sha: str) -> Optional[DocumentRecord]:
        return self._one(
            "SELECT * FROM documents WHERE complete = 1 AND file_sha256 = ?", (sha,)
        )

    def find_by_text_hash(self, sha: str) -> Optional[DocumentRecord]:
        return self._one(
            "SELECT * FROM documents WHERE complete = 1 AND text_sha256 = ?", (sha,)
        )

    def find_by_doi(self, doi: str) -> Optional[DocumentRecord]:
        if not doi:
            return None
        return self._one(
            "SELECT * FROM documents WHERE complete = 1 AND lower(doi) = lower(?)", (doi,)
        )

    def find_by_title_year(self, norm_title: str, year: str) -> Optional[DocumentRecord]:
        if not norm_title or not year:
            return None
        return self._one(
            "SELECT * FROM documents WHERE complete = 1 AND norm_title = ? AND year = ?",
            (norm_title, year),
        )

    def update_details(self, doc_id: str, **values) -> None:
        unknown = set(values) - _EDITABLE
        if unknown:
            raise ValueError(f"Not an editable detail: {', '.join(sorted(unknown))}")
        if not values:
            return
        if "title" in values:
            values["norm_title"] = normalize_title(values["title"])
        assignments = ", ".join(f"{k} = ?" for k in values)
        with self._lock:
            self.connection.execute(
                f"UPDATE documents SET {assignments} WHERE doc_id = ?",
                [*values.values(), doc_id],
            )
            self.connection.commit()

    # -------------------------------------------------------------------------
    # Collections
    # -------------------------------------------------------------------------
    def set_collection(self, doc_id: str, collection: str) -> None:
        with self._lock:
            self.connection.execute(
                "UPDATE documents SET collection = ? WHERE doc_id = ?", (collection, doc_id)
            )
            self.connection.commit()

    def rename_collection(self, old: str, new: str) -> List[str]:
        with self._lock:
            ids = [r[0] for r in self.connection.execute(
                "SELECT doc_id FROM documents WHERE collection = ?", (old,)
            ).fetchall()]
            self.connection.execute(
                "UPDATE documents SET collection = ? WHERE collection = ?", (new, old)
            )
            self.connection.commit()
        return ids

    def collections(self) -> List[str]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT DISTINCT collection FROM documents WHERE complete = 1 ORDER BY collection"
            ).fetchall()
        return [r[0] for r in rows]

    # -------------------------------------------------------------------------
    # Chunks
    # -------------------------------------------------------------------------
    def insert_chunks(self, doc_id: str, chunks: Sequence[Chunk]) -> None:
        parents = {}
        for chunk in chunks:
            parents.setdefault(chunk.parent_id, chunk.parent_text)
        with self._lock:
            self.connection.executemany(
                "INSERT OR IGNORE INTO parents(parent_id, doc_id, text) VALUES (?, ?, ?)",
                [(pid, doc_id, text) for pid, text in parents.items()],
            )
            self.connection.executemany(
                "INSERT INTO chunks(chunk_id, doc_id, parent_id, text, chunk_index, "
                "char_start, char_end, page, slide, section, extra_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.chunk_id, doc_id, c.parent_id, c.text, c.chunk_index,
                     c.char_start, c.char_end, c.page, c.slide, c.section,
                     json.dumps(c.extra))
                    for c in chunks
                ],
            )
            self.connection.commit()

    def load_chunks(self, doc_id: Optional[str] = None) -> List[Chunk]:
        sql = (
            "SELECT c.*, p.text AS parent_text, d.filename, d.doc_type "
            "FROM chunks c "
            "JOIN parents p ON p.parent_id = c.parent_id "
            "JOIN documents d ON d.doc_id = c.doc_id "
        )
        params: List = []
        if doc_id is not None:
            sql += "WHERE c.doc_id = ? "
            params.append(doc_id)
        sql += "ORDER BY d.added_at, c.doc_id, c.chunk_index"
        with self._lock:
            rows = self.connection.execute(sql, params).fetchall()
        return [
            Chunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                parent_id=r["parent_id"],
                parent_text=r["parent_text"],
                source_file=r["filename"],
                doc_type=r["doc_type"],
                chunk_index=r["chunk_index"],
                char_start=r["char_start"],
                char_end=r["char_end"],
                page=r["page"],
                slide=r["slide"],
                section=r["section"],
                extra=json.loads(r["extra_json"] or "{}"),
            )
            for r in rows
        ]

    def chunk_ids(self, doc_id: str) -> List[str]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
            ).fetchall()
        return [r[0] for r in rows]

    def update_chunk_extra(self, chunks: Sequence[Chunk]) -> None:
        """Persist changed `extra` dicts (e.g. a new collection label)."""
        with self._lock:
            self.connection.executemany(
                "UPDATE chunks SET extra_json = ? WHERE chunk_id = ?",
                [(json.dumps(c.extra), c.chunk_id) for c in chunks],
            )
            self.connection.commit()

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self.connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self.connection.commit()
