"""
Ingest-time enrichment.

A chunk pulled out of a long document loses the context that made it
meaningful. "The value rose to 12 percent in the second cohort" is unretrievable
for a question about drug efficacy if the drug's name appeared three pages
earlier. Contextual headers fix that by writing the missing context back into
the chunk before it is embedded.

This is the most expensive option in the app: one model call per chunk. In an
app whose index dies with the process, that cost is paid again every session,
which is why it is off by default and always shows an estimate first.
"""

import re
from typing import Callable, Dict, List, Optional, Sequence

from .chunker import Chunk

# Enough of a document to characterize it without spending a whole context
# window on the summarization step.
_SUMMARY_SOURCE_CHARS = 6000


def _clean(text: str, max_chars: int) -> Optional[str]:
    """Trim a model response down to usable prose, or None."""
    if not text:
        return None

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = " ".join(line.strip() for line in text.split("\n") if line.strip())
    text = text.strip().strip('"\'` ')

    # Drop a leading "Here is the context:" style lead-in.
    text = re.sub(
        r"^(sure[,!]?\s*)?(here('s| is)|this (chunk|passage|section))[^:.]{0,60}:\s*",
        "", text, flags=re.IGNORECASE,
    ).strip()

    if len(text) < 10:
        return None
    return text[:max_chars]


def summarize_document(
    blocks: Sequence[Dict],
    llm: Callable[[str], str],
    max_chars: int = 400,
) -> Optional[str]:
    """
    Write a short characterization of a document.

    One call per *document* rather than per chunk, so this is cheap. The result
    feeds contextual headers and gives filter inference something to match on.

    Args:
        blocks: Extracted blocks for one document
        llm: Callable taking a prompt and returning the model's text
        max_chars: Cap on the summary length

    Returns:
        A summary, or None if the model produced nothing usable
    """
    sample = " ".join(str(b.get("text", "")) for b in blocks)[:_SUMMARY_SOURCE_CHARS]
    if not sample.strip():
        return None

    prompt = (
        "Summarize what this document is about in two sentences. Name its subject, "
        "its type (research paper, report, slide deck, manual), and the main topics "
        "it covers. Output only the summary.\n\n"
        f"Document beginning:\n{sample}\n\n"
        "Summary:"
    )

    try:
        return _clean(llm(prompt), max_chars)
    except Exception:
        return None


def add_contextual_headers(
    chunks: Sequence[Chunk],
    llm: Callable[[str], str],
    document_summaries: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_header_chars: int = 300,
) -> int:
    """
    Attach a situating sentence to each chunk, used at embedding time.

    The header is stored in `chunk.extra['context_header']`, which the store
    prepends before embedding. The chunk's own text is left untouched, so what
    the model reads as evidence and what the user sees as a citation are still
    the document's own words -- only the *vector* is enriched.

    Args:
        chunks: Chunks to enrich, modified in place
        llm: Callable taking a prompt and returning the model's text
        document_summaries: Per-filename summaries from summarize_document
        progress_callback: Called as (completed, total)
        max_header_chars: Cap on each header

    Returns:
        Number of chunks that received a header
    """
    summaries = document_summaries or {}
    enriched = 0
    total = len(chunks)

    for position, chunk in enumerate(chunks, start=1):
        summary = summaries.get(chunk.source_file, "")
        location = chunk.location_label

        prompt = (
            "Write one short sentence situating the excerpt within its document, "
            "so it can be understood on its own. Name the specific subject the "
            "excerpt refers to, replacing vague references. Do not restate the "
            "excerpt and do not add facts. Output only the sentence.\n\n"
            + (f"Document summary: {summary}\n" if summary else "")
            + f"Location: {location}\n\n"
            f"Excerpt:\n{chunk.text[:1500]}\n\n"
            "Sentence:"
        )

        try:
            header = _clean(llm(prompt), max_header_chars)
        except Exception:
            header = None

        if header:
            chunk.extra["context_header"] = header
            enriched += 1

        if progress_callback:
            progress_callback(position, total)

    return enriched


def estimate_enrichment_calls(chunk_count: int, document_count: int) -> int:
    """
    Model calls a full enrichment pass would cost.

    Shown before the user commits, because on local hardware this is the
    difference between a few seconds and several minutes.
    """
    return chunk_count + document_count
