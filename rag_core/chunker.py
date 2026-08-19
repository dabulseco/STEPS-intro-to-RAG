"""
Structure-aware chunking.

Two sizes are produced for every piece of text:

  * a **child** chunk (~400 tokens) -- small enough that its embedding
    represents one idea, which is what makes vector matching precise;
  * a **parent** window (~1500 tokens) -- the surrounding block, which is what
    actually reaches the model so it reads complete context rather than a
    fragment cut mid-argument.

Splitting always happens *within* a block, so a chunk never straddles two pages
or two slides and its citation is unambiguous.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .config import (
    CHILD_CHUNK_CHARS,
    CHILD_CHUNK_OVERLAP,
    MIN_CHUNK_CHARS,
    PARENT_MAX_CHARS,
)

# Preferred split points, strongest first. Splitting on a paragraph break keeps
# a chunk coherent; splitting mid-word is the last resort.
_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


@dataclass
class Chunk:
    """One retrievable unit of a document."""

    chunk_id: str
    text: str                       # child text -- embedded and matched
    parent_id: str
    parent_text: str                # what the LLM reads
    source_file: str
    doc_type: str
    chunk_index: int
    char_start: int
    char_end: int
    page: Optional[int] = None
    slide: Optional[int] = None
    section: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def location_label(self) -> str:
        """Short human-readable origin, e.g. 'report.pdf p.4 — Methods'."""
        parts = [self.source_file]
        if self.page:
            parts.append(f"p.{self.page}")
        elif self.slide:
            parts.append(f"slide {self.slide}")
        if self.section:
            parts.append(f"— {self.section}")
        return " ".join(parts)

    def to_metadata(self) -> Dict[str, object]:
        """
        Flatten to Chroma-compatible metadata.

        Chroma stores only scalars and rejects None, so absent page/slide
        numbers become 0 and an absent section becomes "". That keeps every key
        present with a uniform type, which is what `where` filters need.
        """
        metadata = {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "source_file": self.source_file,
            "doc_type": self.doc_type,
            "section": self.section or "",
            "page": self.page or 0,
            "slide": self.slide or 0,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "word_count": len(self.text.split()),
            "location_label": self.location_label,
        }
        for key, value in self.extra.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        return metadata


def _document_id(doc_key: str) -> str:
    """Stable short id derived from a key that uniquely identifies a document."""
    return hashlib.sha1(doc_key.encode("utf-8")).hexdigest()[:8]


def _split_text(text: str, size: int, overlap: int) -> List[tuple]:
    """
    Split text into overlapping windows that end on natural boundaries.

    Returns:
        List of (start_offset, end_offset) pairs into `text`.
    """
    if len(text) <= size:
        return [(0, len(text))]

    spans = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            # Back off to the strongest separator available in the final third
            # of the window, so chunks end at a real boundary without shrinking
            # too far below the target size.
            window_floor = start + int(size * 0.6)
            for separator in _SEPARATORS:
                cut = text.rfind(separator, window_floor, end)
                if cut > start:
                    end = cut + len(separator)
                    break

        spans.append((start, end))

        if end >= len(text):
            break

        # Overlap preserves context across the seam so a sentence split down
        # the middle is still retrievable from one side or the other.
        start = max(end - overlap, start + 1)

    return spans


def _parent_windows(text: str, max_chars: int) -> List[tuple]:
    """Split an oversized block into parent-sized windows (no overlap)."""
    if len(text) <= max_chars:
        return [(0, len(text))]
    return _split_text(text, max_chars, overlap=0)


def chunk_document(
    blocks: Sequence[Dict],
    source_file: str,
    doc_type: str,
    child_chars: int = CHILD_CHUNK_CHARS,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
    parent_chars: int = PARENT_MAX_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
    doc_key: Optional[str] = None,
) -> List[Chunk]:
    """
    Turn extracted blocks into child chunks with parent windows attached.

    Args:
        blocks: Blocks from DocumentProcessor.extract_blocks_from_*
        source_file: Filename, used for citations and metadata filtering
        doc_type: Extension such as '.pdf'
        child_chars: Target size of the embedded unit
        child_overlap: Overlap between consecutive children
        parent_chars: Cap on the context window returned to the LLM
        min_chars: Fragments shorter than this are dropped as noise
        doc_key: Overrides the id basis when two uploads share a filename;
            defaults to the filename itself

    Returns:
        Chunks in document order
    """
    doc_id = _document_id(doc_key or source_file)
    chunks: List[Chunk] = []
    doc_offset = 0      # running character position across the whole document
    parent_counter = 0

    for block in blocks:
        block_text = str(block.get("text", "")).strip()
        if not block_text:
            continue

        page = block.get("page")
        slide = block.get("slide")
        section = block.get("section")

        for p_start, p_end in _parent_windows(block_text, parent_chars):
            parent_text = block_text[p_start:p_end]
            parent_id = f"{doc_id}:p{parent_counter}"
            parent_counter += 1

            child_spans = _split_text(parent_text, child_chars, child_overlap)

            for c_start, c_end in child_spans:
                child_text = parent_text[c_start:c_end].strip()

                # Drop noise fragments, but never drop a parent's only child --
                # a short table or a one-line slide is still worth retrieving.
                if len(child_text) < min_chars and len(child_spans) > 1:
                    continue
                if not child_text:
                    continue

                absolute_start = doc_offset + p_start + c_start
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}:c{len(chunks)}",
                    text=child_text,
                    parent_id=parent_id,
                    parent_text=parent_text,
                    source_file=source_file,
                    doc_type=doc_type,
                    chunk_index=len(chunks),
                    char_start=absolute_start,
                    char_end=absolute_start + len(child_text),
                    page=page,
                    slide=slide,
                    section=section,
                ))

        doc_offset += len(block_text)

    return chunks


def chunk_documents(results: Sequence[Dict]) -> List[Chunk]:
    """
    Chunk a batch of processed documents.

    Args:
        results: Dicts from DocumentProcessor.process_document, each carrying
            'blocks', 'filename' and 'file_type'

    Returns:
        Chunks across all documents that extracted successfully
    """
    all_chunks: List[Chunk] = []

    for position, result in enumerate(results):
        if result.get("error") or not result.get("blocks"):
            continue
        all_chunks.extend(chunk_document(
            blocks=result["blocks"],
            source_file=result["filename"],
            doc_type=result.get("file_type") or "",
            # Two uploads can share a filename; the position disambiguates them
            # so their chunk ids cannot collide and overwrite each other.
            doc_key=f"{position}:{result['filename']}",
        ))

    return all_chunks
