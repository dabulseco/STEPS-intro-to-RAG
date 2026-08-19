"""
rag_core -- a storage- and UI-agnostic retrieval engine.

Deliberately free of Streamlit and of any assumption about how long the vector
store lives, so it can be lifted wholesale into a persistent application by
swapping the store's Chroma client.
"""

from .config import OLLAMA_BASE_URL
from .embeddings import EmbeddingError, OllamaEmbedder
from .ollama_api import (
    OllamaUnavailable,
    list_chat_models,
    list_embedding_models,
    list_models,
)

__all__ = [
    "OLLAMA_BASE_URL",
    "OllamaEmbedder",
    "EmbeddingError",
    "OllamaUnavailable",
    "list_models",
    "list_chat_models",
    "list_embedding_models",
]
