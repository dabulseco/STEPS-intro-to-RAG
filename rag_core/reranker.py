"""
Re-ranking: a second, more expensive look at the candidates.

Retrieval optimizes for recall -- cast a wide net and accept some noise.
Re-ranking optimizes for precision: score each surviving candidate against the
question directly, then keep only the best. The two stages want opposite
things, which is exactly why doing both beats doing either harder.

Two interchangeable backends:

  * **LLMReranker** (default) scores every candidate in a *single* call, so the
    cost is one extra inference per question rather than one per candidate.
  * **CrossEncoderReranker** is faster and more accurate per query but pulls in
    sentence-transformers and downloads a model, so it stays opt-in.

Both fail soft: if scoring does not work, the original retrieval order is
returned unchanged. A broken reranker must never be worse than no reranker.
"""

import json
import re
from typing import Callable, List, Optional, Sequence

from .store import RetrievedChunk

# Enough of a chunk to judge relevance without spending the whole context
# window on candidates that are about to be discarded.
_SNIPPET_CHARS = 500


class LLMReranker:
    """
    Scores candidates with the chat model already in use.

    Args:
        llm: Callable taking a prompt and returning the model's text
        snippet_chars: How much of each candidate to show the judge
    """

    def __init__(self, llm: Callable[[str], str], snippet_chars: int = _SNIPPET_CHARS):
        self.llm = llm
        self.snippet_chars = snippet_chars

    def __call__(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """
        Reorder candidates by judged relevance.

        Args:
            query: The search query
            candidates: Retrieved chunks to judge
            top_k: How many to return

        Returns:
            Up to top_k chunks, most relevant first
        """
        if not candidates:
            return []

        scores = self._score(query, candidates)
        if not scores:
            # Scoring failed; retrieval order is still a reasonable ordering.
            return list(candidates)[:top_k]

        for position, hit in enumerate(candidates):
            hit.rerank_score = scores.get(position)

        # Unscored candidates sort last but keep their relative retrieval order.
        ranked = sorted(
            enumerate(candidates),
            key=lambda pair: (
                pair[1].rerank_score if pair[1].rerank_score is not None else -1.0,
                -pair[0],
            ),
            reverse=True,
        )
        return [hit for _, hit in ranked][:top_k]

    def _score(self, query: str, candidates: Sequence[RetrievedChunk]) -> dict:
        """Ask the model to score every candidate in one call."""
        listing = "\n\n".join(
            f"[{i}] {hit.chunk.location_label}\n"
            f"{hit.chunk.text[:self.snippet_chars]}"
            for i, hit in enumerate(candidates)
        )

        prompt = (
            "Rate how well each passage helps answer the question. Use a scale of "
            "0 to 10, where 0 means irrelevant and 10 means it directly answers the "
            "question. Judge only relevance to this question, not writing quality.\n\n"
            f"Question: {query}\n\n"
            f"Passages:\n{listing}\n\n"
            "Respond with a JSON object mapping each passage number to its score, "
            'for example {"0": 8, "1": 2}. Output only the JSON.'
        )

        try:
            raw = self.llm(prompt) or ""
        except Exception:
            return {}

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
        except (ValueError, TypeError):
            return {}

        if not isinstance(parsed, dict):
            return {}

        scores = {}
        for key, value in parsed.items():
            try:
                index = int(str(key).strip().strip("[]"))
                score = float(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(candidates):
                # Clamp: models occasionally answer on a different scale.
                scores[index] = max(0.0, min(10.0, score))

        return scores


class CrossEncoderReranker:
    """
    Scores candidates with a cross-encoder model.

    Unlike the bi-encoder used for retrieval, a cross-encoder reads the question
    and the passage together, which is what makes it more accurate -- and why it
    cannot be precomputed into an index.

    Requires `pip install sentence-transformers` and downloads a model on first
    use, so construction raises if the dependency is absent.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker needs sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def __call__(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Reorder candidates by cross-encoder score."""
        if not candidates:
            return []

        pairs = [(query, hit.chunk.text[:_SNIPPET_CHARS]) for hit in candidates]

        try:
            scores = self._model.predict(pairs)
        except Exception:
            return list(candidates)[:top_k]

        for hit, score in zip(candidates, scores):
            hit.rerank_score = float(score)

        ranked = sorted(candidates, key=lambda h: h.rerank_score or 0.0, reverse=True)
        return list(ranked)[:top_k]


def build_reranker(
    backend: str,
    llm: Optional[Callable[[str], str]] = None,
) -> Optional[Callable]:
    """
    Construct a reranker, or None when re-ranking is disabled.

    Args:
        backend: 'llm', 'cross-encoder' or 'none'
        llm: Required for the 'llm' backend

    Returns:
        A callable (query, candidates, top_k) -> reordered candidates
    """
    if backend == "llm" and llm is not None:
        return LLMReranker(llm)
    if backend == "cross-encoder":
        return CrossEncoderReranker()
    return None
