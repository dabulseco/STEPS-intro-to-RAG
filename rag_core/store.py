"""
Ephemeral hybrid retrieval store.

Holds two indexes over the same chunk list:

  * **dense** -- Chroma's in-memory client, for semantic similarity;
  * **sparse** -- BM25, for exact term matching.

Neither alone is sufficient. Dense retrieval misses rare literal tokens (gene
names, catalog numbers, statute references) because they carry little semantic
signal; BM25 misses paraphrases that share no vocabulary with the question.
Their rankings are combined with Reciprocal Rank Fusion.

Nothing is written to disk. The store lives as long as the process and can be
emptied at any time with `clear()`. Making it persistent is a one-line change
to `_new_client`.
"""

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import chromadb
from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .config import (
    COLLECTION_NAME,
    DENSE_TOP_K,
    DENSE_WEIGHT,
    DISTANCE_METRIC,
    FUSED_TOP_K,
    RRF_K,
    SPARSE_TOP_K,
)
from .embeddings import OllamaEmbedder

# Keeps alphanumeric runs together with internal hyphens, dots and slashes, so
# "3.2-fold", "CFTR-KO" and "ms-marco/v2" survive as single searchable tokens
# instead of being shattered into meaningless pieces.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")

# BM25's IDF term down-weights common words but does not neutralize them, and
# its length normalization then favors short passages that happen to contain
# several of them. A naturally-phrased question ("what happens when a cell
# decides to destroy itself") is mostly function words, so without this the
# sparse index ranks on grammar rather than on content.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself me more most my myself no nor
not of off on once only or other ought our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why with would you your yours yourself yourselves
""".split())


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Normalize text into BM25 tokens.

    Stopwords are dropped from both the corpus and the query. If a query is
    *entirely* stopwords the caller gets an empty token list and the sparse
    index simply abstains, leaving the dense index to answer.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


@dataclass
class RetrievedChunk:
    """A chunk plus the evidence for why it was retrieved."""

    chunk: Chunk
    fused_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @property
    def retrieval_source(self) -> str:
        """Which index found this -- shown in the diagnostics panel."""
        if self.dense_rank is not None and self.sparse_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "semantic"
        if self.sparse_rank is not None:
            return "keyword"
        return "unknown"


class EphemeralRagStore:
    """
    In-memory hybrid index over document chunks.

    The embedding model is fixed for the lifetime of an index: embedding
    documents with one model and querying with another silently destroys
    similarity, so changing models requires a rebuild.
    """

    def __init__(
        self,
        embedder: OllamaEmbedder,
        collection_name: str = COLLECTION_NAME,
        distance_metric: str = DISTANCE_METRIC,
    ):
        self.embedder = embedder

        # chromadb.EphemeralClient() hands back a process-wide singleton, so two
        # stores alive at once (which happens the moment the user switches
        # embedding model) would collide on a fixed collection name. A unique
        # suffix per instance keeps them independent.
        self.collection_name = f"{collection_name}_{uuid4().hex[:12]}"
        self.distance_metric = distance_metric

        self._client = self._new_client()
        self._collection = self._new_collection()

        self._chunks: Dict[str, Chunk] = {}
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[str] = []

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def _new_client(self):
        # The single line that decides persistence. Swapping in
        # chromadb.PersistentClient(path=...) is all a persistent app needs.
        return chromadb.EphemeralClient()

    def _new_collection(self):
        # embedding_function=None is load-bearing: Chroma's default function
        # downloads an ONNX model on first use, which would break the promise
        # that nothing but Ollama is ever contacted.
        return self._client.create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": self.distance_metric},
        )

    def clear(self) -> None:
        """Drop every index. Leaves the store usable for a fresh ingest."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            # Already gone; recreating below is still correct.
            pass

        self._collection = self._new_collection()
        self._chunks = {}
        self._bm25 = None
        self._bm25_ids = []

    def close(self) -> None:
        """
        Release the underlying collection for good.

        Chroma's ephemeral client is shared across the process, so a discarded
        store would otherwise keep its vectors resident for the life of the app.
        Unlike clear(), this does not leave the store usable.
        """
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._chunks = {}
        self._bm25 = None
        self._bm25_ids = []

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self._chunks

    @property
    def count(self) -> int:
        return len(self._chunks)

    @property
    def source_files(self) -> List[str]:
        """Distinct filenames currently indexed, in alphabetical order."""
        return sorted({c.source_file for c in self._chunks.values()})

    def stats(self) -> Dict[str, object]:
        """Summary for the UI."""
        per_file: Dict[str, int] = {}
        for chunk in self._chunks.values():
            per_file[chunk.source_file] = per_file.get(chunk.source_file, 0) + 1

        return {
            "chunks": self.count,
            "documents": len(per_file),
            "per_file": per_file,
            "embed_model": self.embedder.model,
            "dimension": self.embedder.dimension,
        }

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def all_chunks(self, source_file: Optional[str] = None) -> List[Chunk]:
        """Indexed chunks in document order, optionally limited to one file."""
        chunks = [
            c for c in self._chunks.values()
            if source_file is None or c.source_file == source_file
        ]
        return sorted(chunks, key=lambda c: (c.source_file, c.chunk_index))

    # -------------------------------------------------------------------------
    # Ingest
    # -------------------------------------------------------------------------
    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Embed and index chunks into both the dense and sparse indexes.

        Args:
            chunks: Chunks to add
            progress_callback: Called as (completed, total) during embedding

        Returns:
            Number of chunks actually added
        """
        new_chunks = [c for c in chunks if c.chunk_id not in self._chunks]
        if not new_chunks:
            return 0

        texts = [self._embed_text(c) for c in new_chunks]
        embeddings = self.embedder.embed_documents(texts, progress_callback=progress_callback)

        self._collection.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=embeddings,
            documents=[c.text for c in new_chunks],
            metadatas=[c.to_metadata() for c in new_chunks],
        )

        for chunk in new_chunks:
            self._chunks[chunk.chunk_id] = chunk

        self._rebuild_bm25()
        return len(new_chunks)

    @staticmethod
    def _embed_text(chunk: Chunk) -> str:
        """
        Text handed to the embedding model.

        A contextual header (Phase 6 enrichment) is prepended when present so
        the vector reflects the chunk's place in its document, not just its
        isolated wording.
        """
        header = chunk.extra.get("context_header")
        return f"{header}\n\n{chunk.text}" if header else chunk.text

    def _rebuild_bm25(self) -> None:
        """
        Rebuild the sparse index from scratch.

        Cheap because the corpus is session-scoped, and it keeps BM25 exactly in
        step with the dense index -- the reconciliation problem that makes
        hybrid search painful in persistent systems simply does not arise here.
        """
        self._bm25_ids = list(self._chunks.keys())
        corpus = [tokenize(self._chunks[cid].text) for cid in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    def _ids_matching(self, where: Optional[Dict]) -> Optional[set]:
        """Chunk ids satisfying a metadata filter, or None when unfiltered."""
        if not where:
            return None
        try:
            result = self._collection.get(where=where, include=[])
            return set(result.get("ids", []))
        except Exception:
            # A malformed filter should degrade to unfiltered search rather
            # than failing the user's query outright.
            return None

    def dense_search(
        self,
        query_embedding: Sequence[float],
        k: int = DENSE_TOP_K,
        where: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """Vector search. Returns (chunk_id, similarity) best-first."""
        if self.is_empty:
            return []

        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(k, self.count),
            where=where or None,
        )

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]

        # Chroma returns cosine *distance*; similarity is the useful direction.
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]

    def sparse_search(
        self,
        query: str,
        k: int = SPARSE_TOP_K,
        allowed_ids: Optional[set] = None,
    ) -> List[Tuple[str, float]]:
        """BM25 search. Returns (chunk_id, score) best-first."""
        if self._bm25 is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda p: p[1], reverse=True)

        # BM25 has no native metadata filtering, so the filter is applied to its
        # output. Chroma filters natively, which is why only this side needs it.
        if allowed_ids is not None:
            ranked = [(cid, s) for cid, s in ranked if cid in allowed_ids]

        # A zero score means no query term occurs in the chunk at all.
        return [(cid, float(s)) for cid, s in ranked[:k] if s > 0]

    def search(
        self,
        query: str,
        query_embedding: Optional[Sequence[float]] = None,
        dense_k: int = DENSE_TOP_K,
        sparse_k: int = SPARSE_TOP_K,
        fused_k: int = FUSED_TOP_K,
        dense_weight: float = DENSE_WEIGHT,
        where: Optional[Dict] = None,
        rrf_k: int = RRF_K,
    ) -> List[RetrievedChunk]:
        """
        Hybrid search over both indexes, fused with RRF.

        Args:
            query: Query text (used for BM25, and embedded if no vector given)
            query_embedding: Precomputed query vector, if one is already at hand
            dense_k / sparse_k: Candidates drawn from each index
            fused_k: Survivors returned after fusion
            dense_weight: 0.0 = keyword only, 1.0 = semantic only
            where: Chroma metadata filter applied before searching
            rrf_k: RRF damping constant

        Returns:
            Retrieved chunks, best-first, carrying per-index ranks and scores
        """
        if self.is_empty:
            return []

        if query_embedding is None:
            query_embedding = self.embedder.embed_query(query)

        allowed_ids = self._ids_matching(where)

        dense_hits = self.dense_search(query_embedding, k=dense_k, where=where)
        sparse_hits = self.sparse_search(query, k=sparse_k, allowed_ids=allowed_ids)

        return self._fuse(dense_hits, sparse_hits, fused_k, dense_weight, rrf_k)

    def _fuse(
        self,
        dense_hits: List[Tuple[str, float]],
        sparse_hits: List[Tuple[str, float]],
        fused_k: int,
        dense_weight: float,
        rrf_k: int,
    ) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion.

        Fusing *ranks* rather than scores is deliberate: a cosine similarity and
        a BM25 score are not on comparable scales, and any normalization that
        made them comparable would need retuning for every corpus. Rank position
        needs none.
        """
        dense_ranks = {cid: i for i, (cid, _) in enumerate(dense_hits)}
        sparse_ranks = {cid: i for i, (cid, _) in enumerate(sparse_hits)}
        dense_scores = dict(dense_hits)
        sparse_scores = dict(sparse_hits)

        sparse_weight = 1.0 - dense_weight
        fused: Dict[str, float] = {}

        for cid, rank in dense_ranks.items():
            fused[cid] = fused.get(cid, 0.0) + dense_weight / (rrf_k + rank + 1)
        for cid, rank in sparse_ranks.items():
            fused[cid] = fused.get(cid, 0.0) + sparse_weight / (rrf_k + rank + 1)

        ordered = sorted(fused.items(), key=lambda p: p[1], reverse=True)[:fused_k]

        results = []
        for cid, score in ordered:
            chunk = self._chunks.get(cid)
            if chunk is None:
                continue
            results.append(RetrievedChunk(
                chunk=chunk,
                fused_score=score,
                dense_rank=dense_ranks.get(cid),
                sparse_rank=sparse_ranks.get(cid),
                dense_score=dense_scores.get(cid),
                sparse_score=sparse_scores.get(cid),
            ))

        return results
