"""
Test doubles shared by the persistent-library tests.

Nothing here talks to Ollama: embeddings are bag-of-characters vectors and the
LLMs are plain functions, so every library test is fast and deterministic.
"""

import io
import json
from typing import Callable, Dict, List, Sequence, Tuple


class StubEmbedder:
    """
    Deterministic embeddings with no network.

    Same scheme as the one in test_rag_core.py: meaningless semantically, but
    stable and comparable, which is enough to exercise ranking and persistence.
    """

    def __init__(self, model: str = "stub-embed"):
        self.model = model
        self.dimension = 26

    def _vector(self, text):
        vec = [0.0] * 26
        for char in text.lower():
            if "a" <= char <= "z":
                vec[ord(char) - 97] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts, progress_callback=None):
        vectors = [self._vector(t) for t in texts]
        if progress_callback:
            progress_callback(len(texts), len(texts))
        return vectors

    def embed_query(self, text):
        return self._vector(text)


class FakeUpload:
    """Stands in for a Streamlit UploadedFile."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self.size = len(data)

    def read(self) -> bytes:
        return self._data

    def getvalue(self) -> bytes:
        return self._data


# Distinct rare tokens per block: BM25 needs terms that do not occur everywhere.
_RARE = ["zq3x", "vk9w", "pj7y", "xb2m", "qf8n", "wd5t", "hy4r", "gc6s"]


def make_blocks(n: int = 3, prefix: str = "") -> List[Dict]:
    """Page blocks sharing common wording, each carrying a unique rare token."""
    return [
        {
            "text": f"{prefix} The quick brown fox jumps over the lazy dog. "
                    f"Marker token {_RARE[i % len(_RARE)]}{prefix} appears here "
                    f"on page {i + 1} of the study.",
            "page": i + 1,
            "slide": None,
            "section": "Results" if i else "Abstract",
            "block_index": i,
        }
        for i in range(n)
    ]


def processed(filename: str, blocks: List[Dict]) -> Dict:
    """A DocumentProcessor.process_document-shaped result."""
    return {
        "filename": filename,
        "blocks": blocks,
        "content": "\n\n".join(b["text"] for b in blocks),
        "error": None,
        "warning": None,
        "file_type": "." + filename.rsplit(".", 1)[-1],
        "size": 1000,
    }


def details_llm(title="A Study of Foxes", authors="Smith, J.; Lee, K.",
                year="2021", journal="J. Canid Res.", doi="") -> Callable[[str], str]:
    """An ask_llm stub that answers the publication-details prompt with JSON."""
    payload = json.dumps({"title": title, "authors": authors, "year": year,
                          "journal": journal, "doi": doi})

    def ask(prompt: str) -> str:
        return f"```json\n{payload}\n```"

    return ask


def silent_llm(prompt: str) -> str:
    """An ask_llm stub that returns nothing useful."""
    return ""


class RecordingChat:
    """A chat_llm stub that records the messages it received."""

    def __init__(self, reply: str = "Foxes jump over dogs [S1]."):
        self.reply = reply
        self.calls: List[Sequence[Tuple[str, str]]] = []

    def __call__(self, messages):
        self.calls.append(list(messages))
        return self.reply
