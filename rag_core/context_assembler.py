"""
Context assembly: turning retrieved chunks into a prompt block that fits.

This module is where the app's central bug is fixed. Previously every uploaded
document was concatenated and pasted into the prompt regardless of size, so
Ollama silently truncated it and the model answered from whatever fragment
survived. Here the context is packed against an explicit character budget
derived from `num_ctx`, and anything that does not fit is *reported* rather
than silently lost.

Retrieved children are expanded to their parent windows on the way in, so the
model reads coherent passages instead of the narrow slices that matched.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import CHARS_PER_TOKEN, CONTEXT_BUDGET_RATIO
from .store import RetrievedChunk


@dataclass
class Source:
    """One numbered piece of evidence, as the model and the UI both see it."""

    index: int                      # 1-based; rendered as [S1], [S2], ...
    chunk_id: str
    text: str                       # what was actually included in the prompt
    location_label: str
    source_file: str
    page: Optional[int] = None
    slide: Optional[int] = None
    section: Optional[str] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_source: str = "unknown"

    @property
    def marker(self) -> str:
        return f"[S{self.index}]"


@dataclass
class AssembledContext:
    """The context block plus an account of what went into it."""

    text: str
    sources: List[Source]
    used_chars: int
    budget_chars: int
    dropped_count: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.sources

    @property
    def approx_tokens(self) -> int:
        return int(self.used_chars / CHARS_PER_TOKEN)

    @property
    def budget_used_pct(self) -> float:
        if not self.budget_chars:
            return 0.0
        return 100.0 * self.used_chars / self.budget_chars


def context_budget_chars(
    num_ctx: int,
    ratio: float = CONTEXT_BUDGET_RATIO,
    chars_per_token: float = CHARS_PER_TOKEN,
) -> int:
    """
    Characters of retrieved context that fit in the model's window.

    Only a fraction of `num_ctx` is available: the system prompt, the chat
    history, the question itself and the model's own answer all have to live in
    the same window. Reserving the remainder is what keeps the prompt from
    being truncated at the point where the evidence lives.
    """
    return max(0, int(num_ctx * ratio * chars_per_token))


def assemble_context(
    retrieved: Sequence[RetrievedChunk],
    num_ctx: int,
    ratio: float = CONTEXT_BUDGET_RATIO,
    chars_per_token: float = CHARS_PER_TOKEN,
    use_parent_windows: bool = True,
) -> AssembledContext:
    """
    Pack retrieved chunks into a numbered, budget-respecting context block.

    Args:
        retrieved: Chunks in relevance order (best first)
        num_ctx: The model's configured context window, in tokens
        ratio: Share of the window reserved for retrieved context
        chars_per_token: Conversion heuristic
        use_parent_windows: Send the surrounding passage rather than the
            matched fragment alone

    Returns:
        AssembledContext carrying the prompt text and its source table
    """
    budget = context_budget_chars(num_ctx, ratio, chars_per_token)

    sources: List[Source] = []
    parts: List[str] = []
    seen_parents = set()
    used = 0
    dropped = 0

    for hit in retrieved:
        chunk = hit.chunk

        # Two children of the same parent would otherwise duplicate the entire
        # surrounding passage and burn the budget twice over.
        if use_parent_windows and chunk.parent_id in seen_parents:
            continue

        body = chunk.parent_text if use_parent_windows else chunk.text
        header = f"[S{len(sources) + 1}] {chunk.location_label}"
        entry = f"{header}\n{body}"

        if used + len(entry) > budget:
            # Fall back to the narrow matched text -- half the evidence beats
            # none of it -- and only give up if even that will not fit.
            fallback = f"{header}\n{chunk.text}"
            if used + len(fallback) > budget:
                dropped += 1
                continue
            entry, body = fallback, chunk.text

        seen_parents.add(chunk.parent_id)
        parts.append(entry)
        used += len(entry)

        sources.append(Source(
            index=len(sources) + 1,
            chunk_id=chunk.chunk_id,
            text=body,
            location_label=chunk.location_label,
            source_file=chunk.source_file,
            page=chunk.page,
            slide=chunk.slide,
            section=chunk.section,
            fused_score=hit.fused_score,
            rerank_score=hit.rerank_score,
            retrieval_source=hit.retrieval_source,
        ))

    return AssembledContext(
        text="\n\n".join(parts),
        sources=sources,
        used_chars=used,
        budget_chars=budget,
        dropped_count=dropped,
    )


def format_source_list(sources: Sequence[Source]) -> str:
    """Compact source table, e.g. for appending to an export."""
    return "\n".join(f"{s.marker} {s.location_label}" for s in sources)
