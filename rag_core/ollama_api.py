"""
Thin wrapper over the Ollama HTTP API for model discovery.

Kept separate from the embedder so the Streamlit layer can ask what is
available without constructing an embedding client first.
"""

from typing import Dict, List

import requests

from .config import OLLAMA_BASE_URL, TAGS_TIMEOUT

# Embedding models are returned by /api/tags alongside chat models, but they
# cannot generate text -- selecting one as the chat model produces confusing
# failures. They are identifiable either by name or by their BERT-family
# architecture.
_EMBED_FAMILIES = {"bert", "nomic-bert"}
_EMBED_NAME_HINTS = ("embed", "embedding")


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


def list_models(base_url: str = OLLAMA_BASE_URL) -> List[Dict]:
    """
    Fetch all models known to the local Ollama instance.

    Raises:
        OllamaUnavailable: if the server is unreachable or returns an error.
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=TAGS_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaUnavailable(
            f"Could not reach Ollama at {base_url}. Is it running? ({exc})"
        ) from exc

    return response.json().get("models", [])


def is_embedding_model(model: Dict) -> bool:
    """Classify a /api/tags entry as an embedding model."""
    name = model.get("name", "").lower()
    family = (model.get("details") or {}).get("family") or ""

    if any(hint in name for hint in _EMBED_NAME_HINTS):
        return True
    return family.lower() in _EMBED_FAMILIES


def list_embedding_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    """Names of models suitable for embedding, preferred default first."""
    names = [m["name"] for m in list_models(base_url) if is_embedding_model(m)]

    # Surface the fast default ahead of heavier alternatives.
    names.sort(key=lambda n: (not n.startswith("nomic-embed-text"), n))
    return names


def list_chat_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    """Names of models suitable for generation (embedding models excluded)."""
    return [m["name"] for m in list_models(base_url) if not is_embedding_model(m)]
