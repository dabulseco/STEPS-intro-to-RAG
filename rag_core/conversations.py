"""
Saved conversations for persistent libraries.

Each exchange keeps a *copy* of the passages it cited (up to 600 characters),
not a pointer into the index, so an old answer stays readable and checkable
after its source document is removed from the library. Deleting the
conversation or the library removes the copies.

Retrieval diagnostics are deliberately not stored: they are a debugging view
for the current session.
"""

import datetime
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

from .citations import CitationReport
from .library_db import LibraryDB

PASSAGE_CHARS = 600
HISTORY_WINDOW = 5
_TITLE_CHARS = 60
_DEFAULT_TITLE = "New chat"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER NOT NULL
                     REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    position         INTEGER NOT NULL,
    question         TEXT NOT NULL,
    answer           TEXT NOT NULL,
    chat_model       TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    sources_json     TEXT NOT NULL DEFAULT '[]',
    citation_json    TEXT NOT NULL DEFAULT '{}',
    followups_json   TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, position);
"""


def _now() -> str:
    # Microseconds: list order is by last update, and two chats touched in
    # the same second must still sort correctly.
    return datetime.datetime.now().isoformat(timespec="microseconds")


@dataclass
class SavedSource:
    """A cited passage as stored with the answer."""

    marker: str          # "[S1]"
    doc_id: str          # "" when the document could not be resolved
    location_label: str  # "smith2021.pdf p.4 — Methods"
    passage: str         # copy, up to PASSAGE_CHARS


@dataclass
class Exchange:
    message_id: int
    position: int
    question: str
    answer: str
    chat_model: str
    created_at: str
    sources: List[SavedSource] = field(default_factory=list)
    citation_summary: str = ""
    citation_clean: bool = True
    hallucinated: List[int] = field(default_factory=list)
    followups: List[str] = field(default_factory=list)


@dataclass
class ConversationSummary:
    conversation_id: int
    title: str
    created_at: str
    updated_at: str
    exchanges: int


class ConversationStore:
    """Conversations in a library's SQLite file, sharing LibraryDB's connection."""

    def __init__(self, db: LibraryDB):
        self._db = db
        self._conn = db.connection
        with db.lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def create(self, title: str = _DEFAULT_TITLE) -> int:
        now = _now()
        with self._db.lock:
            cur = self._conn.execute(
                "INSERT INTO conversations(title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            self._conn.commit()
            return cur.lastrowid

    def list(self) -> List[ConversationSummary]:
        with self._db.lock:
            rows = self._conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m "
                "             WHERE m.conversation_id = c.conversation_id) AS n "
                "FROM conversations c ORDER BY c.updated_at DESC, c.conversation_id DESC"
            ).fetchall()
        return [
            ConversationSummary(r["conversation_id"], r["title"], r["created_at"],
                                r["updated_at"], r["n"])
            for r in rows
        ]

    def exchanges(self, conversation_id: int) -> List[Exchange]:
        with self._db.lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY position",
                (conversation_id,),
            ).fetchall()
        out = []
        for r in rows:
            citation = json.loads(r["citation_json"] or "{}")
            out.append(Exchange(
                message_id=r["message_id"],
                position=r["position"],
                question=r["question"],
                answer=r["answer"],
                chat_model=r["chat_model"],
                created_at=r["created_at"],
                sources=[SavedSource(**s) for s in json.loads(r["sources_json"] or "[]")],
                citation_summary=citation.get("summary", ""),
                citation_clean=citation.get("clean", True),
                hallucinated=citation.get("hallucinated", []),
                followups=json.loads(r["followups_json"] or "[]"),
            ))
        return out

    def append(
        self,
        conversation_id: int,
        question: str,
        answer: str,
        chat_model: str,
        sources: Sequence[SavedSource],
        citations: Optional[CitationReport],
        followups: Sequence[str],
    ) -> Exchange:
        """Save one finished exchange; the first one names an untitled chat."""
        now = _now()
        stored = [
            SavedSource(s.marker, s.doc_id, s.location_label, (s.passage or "")[:PASSAGE_CHARS])
            for s in sources
        ]
        citation = {}
        if citations is not None:
            citation = {
                "summary": citations.summary(),
                "clean": citations.is_clean,
                "hallucinated": sorted(citations.hallucinated),
            }
        with self._db.lock:
            position = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()[0]
            cur = self._conn.execute(
                "INSERT INTO messages(conversation_id, position, question, answer, chat_model, "
                "created_at, sources_json, citation_json, followups_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, position, question, answer, chat_model, now,
                 json.dumps([asdict(s) for s in stored]), json.dumps(citation),
                 json.dumps(list(followups))),
            )
            if position == 0:
                title = " ".join(question.split())
                if len(title) > _TITLE_CHARS:
                    title = title[:_TITLE_CHARS].rstrip() + "…"
                self._conn.execute(
                    "UPDATE conversations SET title = ? WHERE conversation_id = ? AND title = ?",
                    (title or _DEFAULT_TITLE, conversation_id, _DEFAULT_TITLE),
                )
            self._conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
            self._conn.commit()
            message_id = cur.lastrowid
        return Exchange(
            message_id=message_id, position=position, question=question, answer=answer,
            chat_model=chat_model, created_at=now, sources=stored,
            citation_summary=citation.get("summary", ""),
            citation_clean=citation.get("clean", True),
            hallucinated=citation.get("hallucinated", []),
            followups=list(followups),
        )

    def rename(self, conversation_id: int, title: str) -> None:
        with self._db.lock:
            self._conn.execute(
                "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                (title.strip() or _DEFAULT_TITLE, conversation_id),
            )
            self._conn.commit()

    def delete(self, conversation_id: int) -> None:
        with self._db.lock:
            self._conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
            )
            self._conn.commit()

    def history(self, conversation_id: int, window: int = HISTORY_WINDOW) -> List[Tuple[str, str]]:
        """The last `window` (question, answer) pairs, oldest first."""
        with self._db.lock:
            rows = self._conn.execute(
                "SELECT question, answer FROM messages WHERE conversation_id = ? "
                "ORDER BY position DESC LIMIT ?",
                (conversation_id, window),
            ).fetchall()
        return [(r["question"], r["answer"]) for r in reversed(rows)]
