"""
Shared plumbing for the library app: the open library, Ollama status, and the
LLM callables that `rag_core` expects.

An opened library holds a lock file and an in-memory keyword index, so it is
kept in a process-wide registry (one Streamlit process may serve several
tabs) and only one library is kept open at a time.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from rag_core.config import OLLAMA_BASE_URL
from rag_core.library import ChatbotSettings, Library
from rag_core.ollama_api import OllamaUnavailable, list_chat_models, list_embedding_models


# -----------------------------------------------------------------------------
# Open library registry
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _registry() -> Dict[str, Library]:
    return {}


def get_library(path: Path) -> Library:
    """The open Library for `path`, opening it (and closing any other) if needed."""
    key = str(Path(path).resolve())
    registry = _registry()
    library = registry.get(key)
    if library is not None:
        return library
    close_all()
    library = Library.open(Path(path))
    registry[key] = library
    return library


def forget_library(path: Path) -> None:
    """Close and drop one library (before deleting it, for instance)."""
    library = _registry().pop(str(Path(path).resolve()), None)
    if library is not None:
        library.close()


def close_all() -> None:
    registry = _registry()
    for library in list(registry.values()):
        library.close()
    registry.clear()


# -----------------------------------------------------------------------------
# Ollama
# -----------------------------------------------------------------------------
@dataclass
class OllamaStatus:
    chat_models: List[str] = field(default_factory=list)
    embedding_models: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.error is None


def _base(name: str) -> str:
    return name[:-len(":latest")] if name.endswith(":latest") else name


def model_installed(name: str, installed: List[str]) -> bool:
    """Match 'nomic-embed-text' against 'nomic-embed-text:latest' and vice versa."""
    return bool(name) and _base(name) in {_base(m) for m in installed}


@st.cache_data(ttl=15, show_spinner=False)
def _fetch_status() -> Tuple[List[str], List[str], Optional[str]]:
    try:
        return list_chat_models(), list_embedding_models(), None
    except OllamaUnavailable as e:
        return [], [], str(e)
    except Exception as e:  # noqa: BLE001 -- UI boundary
        return [], [], f"Could not reach Ollama: {e}"


def ollama_status() -> OllamaStatus:
    chats, embeds, error = _fetch_status()
    return OllamaStatus(chats, embeds, error)


# -----------------------------------------------------------------------------
# LLM callables for rag_core
# -----------------------------------------------------------------------------
def make_llms(settings: ChatbotSettings):
    """(chat_llm, ask_llm) bound to the library's chat model and sampling settings."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    engine = ChatOllama(
        model=settings.chat_model,
        base_url=OLLAMA_BASE_URL,
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        num_ctx=settings.num_ctx,
    )
    roles = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}

    def _text(result) -> str:
        return result.content if hasattr(result, "content") else str(result)

    def chat_llm(messages) -> str:
        return _text(engine.invoke([roles[role](content=text) for role, text in messages]))

    def ask_llm(prompt: str) -> str:
        return _text(engine.invoke([HumanMessage(content=prompt)]))

    return chat_llm, ask_llm


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"
