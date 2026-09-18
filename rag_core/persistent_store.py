"""
On-disk hybrid retrieval store for persistent libraries.

Subclasses `EphemeralRagStore` and overrides only where the vectors live, so
search, fusion and filtering behave identically in both apps and
`rag_core.pipeline` accepts either store.

What persists where:
  * vectors + chunk metadata -- Chroma `PersistentClient`, in the library's
    `chroma/` folder;
  * full chunks (incl. parent passages) -- SQLite, via `rag_core.library_db`;
  * BM25 -- nowhere. It is rebuilt in memory from the SQLite chunks when a
    library opens, which takes seconds at the target library size.
"""

from pathlib import Path
from typing import List, Sequence

import chromadb

from .chunker import Chunk
from .config import DISTANCE_METRIC
from .store import EphemeralRagStore


def release_chroma() -> None:
    """
    Drop Chroma's process-wide client cache.

    Chroma keeps one shared system per path for the life of the process; a
    library folder deleted from disk must be released first, or recreating a
    library at the same path would talk to a stale in-memory system.
    """
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


class PersistentRagStore(EphemeralRagStore):
    """Hybrid store whose dense index survives the process."""

    def __init__(
        self,
        embedder,
        chroma_path: Path,
        collection_name: str = "chunks_v1",
        distance_metric: str = DISTANCE_METRIC,
    ):
        # Set before super().__init__, which calls _new_client/_new_collection.
        self._chroma_path = Path(chroma_path)
        self._fixed_name = collection_name
        super().__init__(embedder, collection_name=collection_name,
                         distance_metric=distance_metric)
        # The parent appends a random suffix for ephemeral stores; a persistent
        # collection must keep the same name across sessions.
        self.collection_name = self._fixed_name

    # -------------------------------------------------------------------------
    # Lifecycle overrides
    # -------------------------------------------------------------------------
    def _new_client(self):
        self._chroma_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self._chroma_path))

    def _new_collection(self):
        # embedding_function=None: never download Chroma's default ONNX model.
        return self._client.get_or_create_collection(
            name=self._fixed_name,
            embedding_function=None,
            metadata={"hnsw:space": self.distance_metric},
        )

    def close(self) -> None:
        """Release memory. Unlike the ephemeral store, never deletes vectors."""
        self._chunks = {}
        self._bm25 = None
        self._bm25_ids = []

    def drop_collection(self) -> None:
        """Delete this store's Chroma collection (used after a re-embed swap)."""
        try:
            self._client.delete_collection(self._fixed_name)
        except Exception:
            pass
        self.close()

    # -------------------------------------------------------------------------
    # Persistence helpers
    # -------------------------------------------------------------------------
    def load_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Register chunks whose vectors are already on disk, then rebuild BM25."""
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._rebuild_bm25()

    def missing_vector_ids(self, ids: Sequence[str]) -> List[str]:
        """The subset of ids with no vector in Chroma, in input order."""
        if not ids:
            return []
        present = set()
        ids = list(ids)
        # Chroma limits query size; batch the lookup.
        for start in range(0, len(ids), 5000):
            batch = ids[start:start + 5000]
            present.update(self._collection.get(ids=batch, include=[]).get("ids", []))
        return [i for i in ids if i not in present]

    def remove_chunks(self, ids: Sequence[str]) -> None:
        """Delete chunks from the vector index and memory; rebuild BM25."""
        ids = list(ids)
        for start in range(0, len(ids), 5000):
            batch = ids[start:start + 5000]
            if batch:
                self._collection.delete(ids=batch)
        for cid in ids:
            self._chunks.pop(cid, None)
        self._rebuild_bm25()

    def update_chunk_metadata(self, chunks: Sequence[Chunk]) -> None:
        """Replace Chroma metadata from each chunk (e.g. after a collection move)."""
        chunks = list(chunks)
        if not chunks:
            return
        self._collection.update(
            ids=[c.chunk_id for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                self._chunks[chunk.chunk_id] = chunk
