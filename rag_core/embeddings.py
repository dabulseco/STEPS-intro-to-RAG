"""
Ollama-backed embedding client.

Deliberately dependency-free beyond `requests`: the app stays 100% local, and
this module carries no knowledge of Chroma, Streamlit, or the pipeline above it.
"""

from typing import Callable, List, Optional, Sequence

import requests

from .config import (
    DEFAULT_EMBED_MODEL,
    EMBED_BATCH_SIZE,
    EMBED_TIMEOUT,
    OLLAMA_BASE_URL,
)


class EmbeddingError(RuntimeError):
    """Raised when Ollama fails to produce embeddings."""


class OllamaEmbedder:
    """
    Batched embedding client for a local Ollama instance.

    Query and document embeddings go through the same model -- mixing models
    between ingest and retrieval silently destroys similarity, so the model name
    is fixed at construction and recorded alongside the index.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        batch_size: int = EMBED_BATCH_SIZE,
        timeout: int = EMBED_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self._dimension: Optional[int] = None

    # -----------------------------
    # Public API
    # -----------------------------
    def embed_documents(
        self,
        texts: Sequence[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """
        Embed a batch of documents.

        Args:
            texts: Texts to embed.
            progress_callback: Called as (completed, total) after each batch so
                the UI can drive a progress bar without this module importing
                Streamlit.

        Returns:
            One vector per input text, in the same order.
        """
        if not texts:
            return []

        vectors: List[List[float]] = []
        total = len(texts)

        for start in range(0, total, self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.extend(self._embed_batch(batch))

            if progress_callback:
                progress_callback(min(start + self.batch_size, total), total)

        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._embed_batch([text])[0]

    @property
    def dimension(self) -> Optional[int]:
        """Vector width, known only after the first successful call."""
        return self._dimension

    # -----------------------------
    # Internals
    # -----------------------------
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        # Ollama rejects empty strings; substitute a space so positional
        # alignment with the caller's list is preserved.
        payload = {
            "model": self.model,
            "input": [text if text.strip() else " " for text in batch],
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/embed", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise EmbeddingError(
                f"Embedding request failed for model '{self.model}': {exc}"
            ) from exc

        embeddings = response.json().get("embeddings")
        if not embeddings or len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Expected {len(batch)} embeddings from '{self.model}', "
                f"got {len(embeddings) if embeddings else 0}."
            )

        if self._dimension is None:
            self._dimension = len(embeddings[0])

        return embeddings
