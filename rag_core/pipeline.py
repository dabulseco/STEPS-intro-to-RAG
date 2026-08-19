"""
Retrieval pipeline orchestration.

One entry point -- `answer()` -- replaces the two hand-rolled, copy-pasted
prompt-assembly blocks that previously lived in the chat path and the template
path of app.py. Both now call this, so grounding rules cannot drift apart
between them.

Stages, in order:

    condense -> transform -> retrieve (hybrid) -> rerank -> assemble -> generate

Every stage after retrieval is optional and reports what it did, so the UI can
show why a particular answer came out the way it did.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import (
    CONTEXT_BUDGET_RATIO,
    DENSE_TOP_K,
    DENSE_WEIGHT,
    FINAL_TOP_K,
    FUSED_TOP_K,
    SPARSE_TOP_K,
)
from .citations import CitationReport, verify_citations
from .context_assembler import AssembledContext, Source, assemble_context
from .store import EphemeralRagStore, RetrievedChunk

# Answer-source modes, mirroring the app's existing three-way toggle.
MODE_STRICT = "strict"
MODE_HYBRID = "hybrid"
MODE_LLM = "llm"


@dataclass
class RetrievalSettings:
    """Everything tunable about a single query."""

    dense_k: int = DENSE_TOP_K
    sparse_k: int = SPARSE_TOP_K
    fused_k: int = FUSED_TOP_K
    final_k: int = FINAL_TOP_K
    dense_weight: float = DENSE_WEIGHT
    context_ratio: float = CONTEXT_BUDGET_RATIO
    use_parent_windows: bool = True

    # Query-side transformations (Phase 3)
    condense_history: bool = True
    rewrite_query: bool = False
    multi_query: bool = False
    hyde: bool = False
    infer_filter: bool = False

    # Re-ranking (Phase 4)
    rerank: bool = False


@dataclass
class Diagnostics:
    """A record of what each stage actually did, for the UI panel."""

    original_query: str = ""
    condensed_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    query_variants: List[str] = field(default_factory=list)
    hyde_text: Optional[str] = None
    search_query: str = ""
    where_filter: Optional[Dict] = None
    dense_hits: int = 0
    sparse_hits: int = 0
    fused_hits: int = 0
    reranked: bool = False
    sources_used: int = 0
    context_chars: int = 0
    context_budget: int = 0
    dropped_for_budget: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class RagResult:
    """The answer plus the evidence and the trace behind it."""

    answer: str
    sources: List[Source] = field(default_factory=list)
    mode: str = MODE_LLM
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    context: Optional[AssembledContext] = None
    citations: Optional[CitationReport] = None

    @property
    def has_sources(self) -> bool:
        return bool(self.sources)


def resolve_mode(has_index: bool, allow_llm_knowledge: bool) -> str:
    """
    Map the app's document/knowledge toggles onto a grounding mode.

    Args:
        has_index: Whether any documents are indexed
        allow_llm_knowledge: Whether general knowledge is permitted

    Returns:
        MODE_STRICT, MODE_HYBRID or MODE_LLM
    """
    if not has_index:
        return MODE_LLM
    return MODE_HYBRID if allow_llm_knowledge else MODE_STRICT


def retrieve(
    store: EphemeralRagStore,
    query: str,
    settings: RetrievalSettings,
    where: Optional[Dict] = None,
    reranker: Optional[Callable[[str, Sequence[RetrievedChunk], int], List[RetrievedChunk]]] = None,
    diagnostics: Optional[Diagnostics] = None,
    query_variants: Optional[Sequence[str]] = None,
    hyde_text: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Run hybrid retrieval, optionally over several query phrasings.

    Args:
        store: The populated hybrid store
        query: Search query (already condensed/rewritten by the caller)
        settings: Retrieval knobs
        where: Chroma metadata filter applied before search
        reranker: Optional callable (query, candidates, top_k) -> reordered
        diagnostics: Populated in place when supplied
        query_variants: Extra phrasings to fan out over and fuse
        hyde_text: Hypothetical answer to embed instead of the query itself

    Returns:
        Final ranked chunks, at most settings.final_k of them
    """
    if store is None or store.is_empty:
        return []

    queries = [query] + [q for q in (query_variants or []) if q and q != query]

    # Fan-out: each phrasing retrieves independently, then results are merged by
    # best rank achieved under any phrasing. A passage that any one phrasing
    # finds convincingly should not be buried because the others missed it.
    # HyDE replaces only the *dense* query vector. The literal question still
    # drives BM25, because a hypothetical answer invents terms the corpus may
    # not contain and would poison exact matching.
    primary_embedding = None
    if hyde_text:
        try:
            primary_embedding = store.embedder.embed_query(hyde_text)
        except Exception:
            primary_embedding = None

    merged: Dict[str, RetrievedChunk] = {}
    for position, variant in enumerate(queries):
        for hit in store.search(
            variant,
            query_embedding=primary_embedding if position == 0 else None,
            dense_k=settings.dense_k,
            sparse_k=settings.sparse_k,
            fused_k=settings.fused_k,
            dense_weight=settings.dense_weight,
            where=where,
        ):
            existing = merged.get(hit.chunk.chunk_id)
            if existing is None or hit.fused_score > existing.fused_score:
                merged[hit.chunk.chunk_id] = hit

    candidates = sorted(merged.values(), key=lambda h: h.fused_score, reverse=True)
    candidates = candidates[: settings.fused_k]

    if diagnostics is not None:
        diagnostics.dense_hits = sum(1 for c in candidates if c.dense_rank is not None)
        diagnostics.sparse_hits = sum(1 for c in candidates if c.sparse_rank is not None)
        diagnostics.fused_hits = len(candidates)

    if reranker is not None and settings.rerank and candidates:
        candidates = reranker(query, candidates, settings.final_k)
        if diagnostics is not None:
            diagnostics.reranked = True

    return candidates[: settings.final_k]


def build_user_message(question: str, context: Optional[AssembledContext], mode: str) -> str:
    """
    Compose the user turn.

    The grounding *rules* live in the system prompt; this carries only the
    evidence and the question, which keeps the rules from being diluted as the
    conversation grows.

    Args:
        question: The user's question, as they asked it
        context: Assembled evidence, if any
        mode: Grounding mode

    Returns:
        The user message text
    """
    if mode == MODE_LLM or context is None or context.is_empty:
        return question

    return (
        "DOCUMENT EXCERPTS:\n"
        f"{context.text}\n\n"
        "----------------------------------------\n\n"
        f"QUESTION: {question}"
    )


def answer(
    question: str,
    generate: Callable[[str, str, bool], str],
    store: Optional[EphemeralRagStore] = None,
    mode: str = MODE_LLM,
    num_ctx: int = 8192,
    settings: Optional[RetrievalSettings] = None,
    where: Optional[Dict] = None,
    transformer=None,
    reranker: Optional[Callable] = None,
    chat_history: Optional[Sequence[Tuple[str, str]]] = None,
) -> RagResult:
    """
    Answer a question, retrieving evidence first when documents are indexed.

    Args:
        question: The user's question
        generate: Callable (user_message, grounding_mode, has_context) -> answer.
            Supplied by the app so this module stays free of LangChain and
            Streamlit.
        store: Hybrid store, or None when no documents are loaded
        mode: MODE_STRICT, MODE_HYBRID or MODE_LLM
        num_ctx: Model context window, used to size the evidence budget
        settings: Retrieval knobs
        where: Metadata pre-filter
        transformer: Optional QueryTransformer (Phase 3)
        reranker: Optional reranking callable (Phase 4)
        chat_history: Previous (question, answer) pairs, for condensation

    Returns:
        RagResult with the answer, its sources, and a stage-by-stage trace
    """
    settings = settings or RetrievalSettings()
    diag = Diagnostics(original_query=question, search_query=question, where_filter=where)

    has_index = store is not None and not store.is_empty

    # Strict mode promises the answer comes only from the documents. With no
    # index there is nothing to honor that promise with, so it refuses rather
    # than quietly falling back to general knowledge -- the exact substitution
    # the mode exists to prevent.
    if mode == MODE_STRICT and not has_index:
        diag.notes.append("Strict mode with no indexed documents — declined to answer.")
        return RagResult(
            answer=(
                "No documents are indexed, so there is nothing to answer from. "
                "Upload and process a document, or enable 'Allow LLM General "
                "Knowledge' in the sidebar."
            ),
            mode=mode,
            diagnostics=diag,
        )

    # No index -> nothing to retrieve; answer straight from the model.
    if mode == MODE_LLM or not has_index:
        diag.notes.append("No documents indexed — answered from general knowledge.")
        return RagResult(
            answer=generate(question, MODE_LLM, False),
            mode=MODE_LLM,
            diagnostics=diag,
        )

    # --- Query transformation (Phase 3) ---
    search_query = question
    variants: List[str] = []

    if transformer is not None:
        transformed = transformer.transform(
            question,
            chat_history=chat_history or [],
            settings=settings,
        )
        search_query = transformed.search_query
        variants = transformed.variants
        diag.condensed_query = transformed.condensed_query
        diag.rewritten_query = transformed.rewritten_query
        diag.query_variants = variants
        diag.hyde_text = transformed.hyde_text
        diag.notes.extend(transformed.notes)
        if transformed.where_filter and not where:
            where = transformed.where_filter
            diag.where_filter = where

    diag.search_query = search_query

    # --- Retrieval ---
    retrieved = retrieve(
        store=store,
        query=search_query,
        settings=settings,
        where=where,
        reranker=reranker,
        diagnostics=diag,
        query_variants=variants,
        hyde_text=diag.hyde_text,
    )

    # --- Assembly ---
    context = assemble_context(
        retrieved,
        num_ctx=num_ctx,
        ratio=settings.context_ratio,
        use_parent_windows=settings.use_parent_windows,
    )

    diag.sources_used = len(context.sources)
    diag.context_chars = context.used_chars
    diag.context_budget = context.budget_chars
    diag.dropped_for_budget = context.dropped_count

    if context.dropped_count:
        diag.notes.append(
            f"{context.dropped_count} retrieved passage(s) did not fit the context "
            f"budget and were left out. Raise Context Length to include more."
        )

    if context.is_empty:
        diag.notes.append("Retrieval returned nothing for this query.")
        if mode == MODE_STRICT:
            # Strict mode promises not to answer without evidence, so the app
            # says so directly rather than letting the model improvise.
            return RagResult(
                answer="This information is not found in the provided documents.",
                mode=mode,
                diagnostics=diag,
                context=context,
            )

    user_message = build_user_message(question, context, mode)
    answer_text = generate(user_message, mode, not context.is_empty)

    # Verify the citations rather than trusting them: a fabricated [S7] reads
    # exactly as authoritative as a real one.
    citations = verify_citations(answer_text, context.sources)
    if citations.hallucinated:
        diag.notes.append(
            "The answer cites "
            + ", ".join(f"[S{i}]" for i in sorted(citations.hallucinated))
            + ", which was not among the retrieved sources. Treat it with caution."
        )

    return RagResult(
        answer=answer_text,
        sources=context.sources,
        mode=mode,
        diagnostics=diag,
        context=context,
        citations=citations,
    )
