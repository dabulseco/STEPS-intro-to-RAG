"""
Local LLM Interface with Ollama
A Streamlit application for interacting with local LLM models via Ollama.

Features:
- Dynamic perspective and audience settings for contextualized responses
- Question template system with pre-defined scientific/educational prompts
- Chat history with HTML export capabilities
- Configurable model parameters (temperature, top-k, top-p, context length)
"""

import re
import streamlit as st
import html
from typing import List, Tuple, Optional

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import our utility modules
from utils.template_manager import TemplateManager
from utils.prompt_builder import PromptBuilder
from utils.html_export import build_pair_html, build_conversation_html, safe_filename
from utils.document_processor import DocumentProcessor

# Retrieval engine (storage- and UI-agnostic; see ADVANCED_RAG_PLAN.md)
from rag_core.chunker import chunk_documents
from rag_core.config import (
    DEFAULT_EMBED_MODEL,
    DENSE_TOP_K,
    DENSE_WEIGHT,
    FINAL_TOP_K,
    FUSED_TOP_K,
    OLLAMA_BASE_URL,
    SPARSE_TOP_K,
)
from rag_core.embeddings import EmbeddingError, OllamaEmbedder
from rag_core.enrichment import (
    add_contextual_headers,
    estimate_enrichment_calls,
    summarize_document,
)
from rag_core.ollama_api import (
    OllamaUnavailable,
    list_chat_models,
    list_embedding_models,
)
from rag_core.pipeline import RetrievalSettings, answer as rag_answer, resolve_mode
from rag_core.query_transformer import QueryTransformer
from rag_core.reranker import build_reranker
from rag_core.store import EphemeralRagStore

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Local LLM Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Initialize Template Manager
# -----------------------------
@st.cache_resource
def get_template_manager():
    """Initialize and cache the template manager."""
    return TemplateManager()

template_manager = get_template_manager()

# -----------------------------
# Helper Functions
# -----------------------------
def get_local_models() -> List[str]:
    """
    Fetch chat-capable Ollama models from the local instance.

    Embedding models are excluded: they appear in /api/tags but cannot generate
    text, and selecting one as the chat model fails in confusing ways.

    Returns:
        List of model names, or empty list if unavailable
    """
    try:
        return list_chat_models()
    except OllamaUnavailable as e:
        st.error(f"Error fetching models: {e}")
    return []

def get_embedding_models() -> List[str]:
    """
    Fetch embedding-capable Ollama models from the local instance.

    Returns:
        List of model names, or empty list if unavailable
    """
    try:
        return list_embedding_models()
    except OllamaUnavailable:
        # The chat-model fetch above already surfaced the connection error;
        # reporting it twice adds noise without adding information.
        return []

def generate_ai_response(user_input: str, perspective: str, audience: str,
                        llm_engine, chat_history: List[Tuple[str, str]],
                        has_documents: bool = False,
                        grounding_mode: str = None) -> str:
    """
    Generate AI response using the LLM with context-aware system prompt.

    Args:
        user_input: The user turn to send (question, plus evidence when retrieved)
        perspective: Selected perspective (who's writing)
        audience: Selected audience (who's reading)
        llm_engine: Configured ChatOllama instance
        chat_history: Previous conversation history
        has_documents: Whether retrieved context accompanies this turn
        grounding_mode: 'strict', 'hybrid' or 'llm'; controls the grounding
            rules placed in the system prompt

    Returns:
        AI-generated response string
    """
    # Grounding rules live in the system prompt, where they stay in force for
    # the whole exchange instead of being restated in each user turn.
    system_prompt_text = PromptBuilder.build_system_prompt(
        perspective, audience, has_documents, grounding_mode
    )

    # Concrete messages rather than prompt templates: retrieved document text
    # routinely contains braces (JSON, math, code), which a template would try
    # to interpret as variables and fail on.
    messages = [SystemMessage(content=system_prompt_text)]

    # Build conversation history (limit to last 5 exchanges for context window management)
    for past_query, past_response in chat_history[-5:]:
        messages.append(HumanMessage(content=past_query))
        messages.append(AIMessage(content=past_response))

    # Add current user input
    messages.append(HumanMessage(content=user_input))

    return StrOutputParser().invoke(llm_engine.invoke(messages))

def generate_followup_questions(user_input: str, ai_response: str, llm_engine) -> List[str]:
    """
    Generate 3 suggested follow-up questions based on a Q&A pair.

    Args:
        user_input: The user's original question
        ai_response: The AI's response
        llm_engine: Configured ChatOllama instance

    Returns:
        List of up to 3 follow-up question strings
    """
    prompt = (
        "Based on the following question and answer, generate exactly 3 insightful "
        "follow-up questions that would deepen understanding of the topic. "
        "Return only the 3 questions, one per line, numbered 1. 2. 3. "
        "Do not include any other text or explanation.\n\n"
        f"Question: {user_input}\n\n"
        f"Answer: {ai_response}\n\n"
        "Follow-up questions:"
    )
    try:
        result = llm_engine.invoke([HumanMessage(content=prompt)])
        raw = result.content if hasattr(result, "content") else str(result)

        questions = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned:
                questions.append(cleaned)

        return questions[:3]
    except Exception:
        return []

# -----------------------------
# Session State Initialization
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of (user_query, ai_response) tuples

if "perspective" not in st.session_state:
    st.session_state.perspective = "Scientist"

if "audience" not in st.session_state:
    st.session_state.audience = "Scientist"

if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat"

if "selected_template" not in st.session_state:
    st.session_state.selected_template = None

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "document_content" not in st.session_state:
    st.session_state.document_content = ""

if "allow_llm_knowledge" not in st.session_state:
    st.session_state.allow_llm_knowledge = False  # RAG-first: disabled by default (unchecked)

if "followup_questions" not in st.session_state:
    st.session_state.followup_questions = {}  # Dict: chat_history index -> List[str]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None  # Follow-up question clicked by user

if "rag_store" not in st.session_state:
    st.session_state.rag_store = None  # EphemeralRagStore; dies with the process

if "index_stale_reason" not in st.session_state:
    st.session_state.index_stale_reason = None  # Explains why re-ingest is needed

if "answer_sources" not in st.session_state:
    st.session_state.answer_sources = {}  # Dict: chat_history index -> List[Source]

if "answer_diagnostics" not in st.session_state:
    st.session_state.answer_diagnostics = {}  # Dict: chat_history index -> Diagnostics

if "answer_citations" not in st.session_state:
    st.session_state.answer_citations = {}  # Dict: chat_history index -> CitationReport

# -----------------------------
# Vector Store (ephemeral)
# -----------------------------
def index_is_populated() -> bool:
    """
    Whether any document chunks are currently searchable.

    Read directly from session state rather than via get_store(), so it is safe
    to call before the embedding-model selector has run.
    """
    store = st.session_state.get("rag_store")
    return store is not None and not store.is_empty


def get_store():
    """
    Return this session's retrieval store, creating it on first use.

    The embedding model is baked into an index: vectors built with one model are
    meaningless to another. Changing the model therefore discards the index and
    flags that documents must be re-ingested, rather than silently returning
    nonsense similarity scores.

    Returns:
        EphemeralRagStore, or None if no embedding model is available
    """
    embed_model = st.session_state.get("embed_model")
    if not embed_model:
        return None

    store = st.session_state.rag_store

    if store is None:
        st.session_state.rag_store = EphemeralRagStore(OllamaEmbedder(model=embed_model))
    elif store.embedder.model != embed_model:
        had_content = not store.is_empty
        # Release the old collection; Chroma's ephemeral client is shared across
        # the process, so an abandoned store would keep its vectors resident.
        store.close()
        st.session_state.rag_store = EphemeralRagStore(OllamaEmbedder(model=embed_model))
        if had_content:
            st.session_state.index_stale_reason = (
                f"Embedding model changed to `{embed_model}`. "
                f"The previous index was discarded — re-process your documents."
            )

    return st.session_state.rag_store


def ingest_documents(results, llm_engine=None) -> int:
    """
    Chunk processed documents and index them for retrieval.

    Args:
        results: Dicts from DocumentProcessor.process_multiple_documents
        llm_engine: Needed only when contextual enrichment is enabled

    Returns:
        Number of chunks indexed
    """
    store = get_store()
    if store is None:
        return 0

    chunks = chunk_documents(results)
    if not chunks:
        return 0

    # Optional enrichment, before embedding: the header has to be in place when
    # the vector is computed, or it has no effect at all.
    if st.session_state.get("enrich_chunks") and llm_engine is not None:
        def ask_llm(prompt: str) -> str:
            result = llm_engine.invoke([HumanMessage(content=prompt)])
            return result.content if hasattr(result, "content") else str(result)

        total_calls = estimate_enrichment_calls(len(chunks), len(results))
        enrich_progress = st.progress(
            0.0, text=f"Adding context to chunks (0/{total_calls} model calls)..."
        )
        try:
            summaries = {}
            for position, result in enumerate(results, start=1):
                if result.get("error") or not result.get("blocks"):
                    continue
                summary = summarize_document(result["blocks"], ask_llm)
                if summary:
                    summaries[result["filename"]] = summary
                enrich_progress.progress(
                    position / total_calls,
                    text=f"Summarizing documents ({position}/{total_calls} calls)...",
                )

            offset = len(results)

            def on_enrich(done: int, total: int):
                enrich_progress.progress(
                    min((offset + done) / total_calls, 1.0),
                    text=f"Adding context to chunks ({offset + done}/{total_calls} calls)...",
                )

            add_contextual_headers(chunks, ask_llm, summaries, progress_callback=on_enrich)
        finally:
            enrich_progress.empty()

    progress = st.progress(0.0, text=f"Embedding {len(chunks)} chunks...")

    def on_progress(done: int, total: int):
        progress.progress(done / total, text=f"Embedding chunks... {done}/{total}")

    try:
        added = store.add_chunks(chunks, progress_callback=on_progress)
    finally:
        progress.empty()

    return added


def rebuild_index(llm_engine=None) -> int:
    """
    Rebuild the index from the documents currently loaded.

    Chroma supports deleting by id, but rebuilding is simpler and provably
    consistent: the dense index, the BM25 index and the document list are
    regenerated from one source of truth. The corpus is session-scoped, so the
    cost is bounded by what the user uploaded in this sitting.

    Args:
        llm_engine: Needed only when contextual enrichment is enabled

    Returns:
        Number of chunks indexed
    """
    store = get_store()
    if store is None:
        return 0

    store.clear()
    if not st.session_state.uploaded_documents:
        return 0

    return ingest_documents(st.session_state.uploaded_documents, llm_engine)


def run_query(question: str, llm_engine, num_ctx: int) -> None:
    """
    Answer one question through the retrieval pipeline and record the result.

    This is the single answer path for the whole app. The chat input and the
    template form both call it, which is what keeps their grounding behaviour
    identical -- previously each carried its own copy of the prompt-assembly
    logic and they were free to drift apart.

    Args:
        question: The user's question, as they asked it
        llm_engine: Configured ChatOllama instance
        num_ctx: Context window, used to size the evidence budget
    """
    store = get_store()
    has_index = store is not None and not store.is_empty
    mode = resolve_mode(has_index, st.session_state.allow_llm_knowledge)

    history = list(st.session_state.chat_history)

    def generate(user_message: str, grounding_mode: str, has_context: bool) -> str:
        return generate_ai_response(
            user_message,
            st.session_state.perspective,
            st.session_state.audience,
            llm_engine,
            history,
            has_documents=has_context,
            grounding_mode=grounding_mode,
        )

    settings = RetrievalSettings(
        dense_k=st.session_state.get("dense_k", DENSE_TOP_K),
        sparse_k=st.session_state.get("sparse_k", SPARSE_TOP_K),
        fused_k=st.session_state.get("fused_k", FUSED_TOP_K),
        final_k=st.session_state.get("final_k", FINAL_TOP_K),
        dense_weight=st.session_state.get("dense_weight", DENSE_WEIGHT),
        condense_history=st.session_state.get("condense_history", True),
        rewrite_query=st.session_state.get("rewrite_query", False),
        multi_query=st.session_state.get("multi_query", False),
        hyde=st.session_state.get("use_hyde", False),
        infer_filter=st.session_state.get("infer_filter", False),
        rerank=st.session_state.get("use_rerank", False),
    )

    # A plain prompt -> text callable, so rag_core never sees LangChain.
    def ask_llm(prompt: str) -> str:
        result = llm_engine.invoke([HumanMessage(content=prompt)])
        return result.content if hasattr(result, "content") else str(result)

    transformer = None
    if has_index and any([settings.condense_history, settings.rewrite_query,
                          settings.multi_query, settings.hyde, settings.infer_filter]):
        sections = sorted({
            c.section for c in store.all_chunks() if c.section
        }) if settings.infer_filter else []
        transformer = QueryTransformer(
            ask_llm,
            known_files=store.source_files,
            known_sections=sections,
        )

    reranker = None
    if has_index and settings.rerank:
        backend = st.session_state.get("rerank_backend", "llm")
        try:
            reranker = build_reranker(backend, llm=ask_llm)
        except ImportError as e:
            st.warning(f"⚠️ {e} Falling back to no re-ranking.")

    where = None
    selected_files = st.session_state.get("filter_files") or []
    if has_index and selected_files and len(selected_files) < len(store.source_files):
        # Chroma needs $in for a multi-value match; a single equality clause
        # would silently drop every file but the first.
        where = {"source_file": {"$in": list(selected_files)}}

    result = rag_answer(
        question=question,
        generate=generate,
        store=store,
        mode=mode,
        num_ctx=num_ctx,
        settings=settings,
        where=where,
        transformer=transformer,
        reranker=reranker,
        chat_history=history,
    )

    st.session_state.chat_history.append((question, result.answer))
    index = len(st.session_state.chat_history) - 1

    # Sources and diagnostics ride alongside the history rather than inside it,
    # so the existing (query, response) tuple shape -- which the HTML exporters
    # depend on -- stays intact.
    st.session_state.answer_sources[index] = result.sources
    st.session_state.answer_diagnostics[index] = result.diagnostics
    st.session_state.answer_citations[index] = result.citations

    if st.session_state.get("generate_followups", True):
        followups = generate_followup_questions(question, result.answer, llm_engine)
        if followups:
            st.session_state.followup_questions[index] = followups

# -----------------------------
# Navigation
# -----------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Chat", "Manage Templates"],
    key="navigation",
    index=0 if st.session_state.current_page == "Chat" else 1
)
st.session_state.current_page = page

# -----------------------------
# Sidebar: Perspective & Audience (applies to whole conversation)
# -----------------------------
st.sidebar.title("🎭 Context Settings")
st.sidebar.markdown("*These settings apply to the entire conversation*")

perspectives = PromptBuilder.get_available_perspectives()
st.session_state.perspective = st.sidebar.selectbox(
    "Perspective (Who is writing):",
    perspectives,
    index=perspectives.index(st.session_state.perspective),
    help="Select who should be explaining/writing the answers"
)

audiences = PromptBuilder.get_available_audiences()
st.session_state.audience = st.sidebar.selectbox(
    "Audience (Who is reading):",
    audiences,
    index=audiences.index(st.session_state.audience),
    help="Select who the explanations should be tailored for"
)

# -----------------------------
# Sidebar: RAG Mode Control
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.title("🎯 Answer Source")

# These indicators key off the *index*, not off extracted text. Retrieval can
# only ground answers in what was actually indexed, so promising "documents
# only" while the index is empty would be a claim the pipeline cannot keep.
if index_is_populated():
    st.sidebar.success("📄 Documents Indexed - RAG Mode Active")
    st.sidebar.markdown("**Answer source:** Uploaded documents")

    st.session_state.allow_llm_knowledge = st.sidebar.checkbox(
        "☑️ Allow LLM General Knowledge",
        value=st.session_state.allow_llm_knowledge,
        help="Enable this to allow the AI to supplement answers with its general knowledge. "
             "When disabled (default), AI answers ONLY from your uploaded documents."
    )

    if st.session_state.allow_llm_knowledge:
        st.sidebar.warning("⚠️ AI can use general knowledge + documents")
    else:
        st.sidebar.info("✅ AI will answer ONLY from documents")
elif st.session_state.document_content:
    # Text was extracted but never made it into the index -- usually a missing
    # embedding model. Say so, rather than implying the documents are in play.
    st.sidebar.warning("⚠️ Documents extracted but NOT indexed")
    st.sidebar.markdown("**Answer source:** LLM general knowledge")
    st.sidebar.markdown("*Retrieval is unavailable — check the embedding model above.*")
    st.session_state.allow_llm_knowledge = True
else:
    st.sidebar.info("📝 No Documents - LLM Mode")
    st.sidebar.markdown("**Answer source:** LLM general knowledge")
    st.sidebar.markdown("*Upload documents to enable RAG mode*")
    # Automatically enable LLM knowledge when no documents
    st.session_state.allow_llm_knowledge = True

# -----------------------------
# Sidebar: Model Settings
# -----------------------------
st.sidebar.title("⚙️ Model Settings")
model_options = get_local_models()

if not model_options:
    st.sidebar.error("⚠️ No models available. Ensure Ollama is running.")
    selected_model = None
else:
    selected_model = st.sidebar.selectbox(
        "Select Model:",
        model_options,
        index=0,
        help="Choose which LLM model to use"
    )

temperature = st.sidebar.slider(
    "Temperature:",
    0.0, 2.0, 0.3, 0.05,
    help="Controls randomness: lower = more focused, higher = more creative"
)

top_k = st.sidebar.slider(
    "Top-K:",
    1, 100, 40, 1,
    help="Number of highest probability tokens to consider"
)

top_p = st.sidebar.slider(
    "Top-P:",
    0.0, 1.0, 0.9, 0.01,
    help="Nucleus sampling threshold"
)

num_ctx = st.sidebar.slider(
    "Context Length:",
    256, 32768, 8192, 256,
    help="Maximum context window size. Retrieved context is packed to a budget "
         "derived from this value, so raising it lets more evidence reach the model. "
         "Values beyond what your model was trained for will be ignored by Ollama."
)

# -----------------------------
# Sidebar: Embedding Model (retrieval)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.title("🧬 Retrieval Settings")

embedding_options = get_embedding_models()

if not embedding_options:
    st.sidebar.error(
        "⚠️ No embedding model found. Document retrieval is unavailable.\n\n"
        "Install one with:\n"
        "```\nollama pull nomic-embed-text\n```"
    )
    selected_embed_model = None
else:
    selected_embed_model = st.sidebar.selectbox(
        "Embedding Model:",
        embedding_options,
        index=0,
        help="Model used to embed document chunks and queries. Changing this "
             "invalidates the current index -- documents must be re-ingested."
    )

st.session_state.embed_model = selected_embed_model

# -----------------------------
# Sidebar: Vector Store (ephemeral)
# -----------------------------
_store = get_store()

if st.session_state.index_stale_reason:
    st.sidebar.warning(f"⚠️ {st.session_state.index_stale_reason}")

if _store is not None and not _store.is_empty:
    _stats = _store.stats()
    st.sidebar.success(
        f"🧠 **Vector store:** {_stats['chunks']:,} chunks "
        f"from {_stats['documents']} document(s)"
    )
    st.sidebar.caption(
        f"In memory only — cleared when you press the button below or close the app."
    )

    if st.sidebar.button("🗑️ Clear Vector Store", use_container_width=True):
        _store.clear()
        st.session_state.uploaded_documents = []
        st.session_state.document_content = ""
        st.session_state.index_stale_reason = None
        st.rerun()
elif _store is not None:
    st.sidebar.info("🧠 **Vector store:** empty — upload documents to populate it")

# -----------------------------
# Sidebar: Retrieval Tuning
# -----------------------------
if _store is not None and not _store.is_empty:
    with st.sidebar.expander("🎛️ Retrieval Tuning", expanded=False):
        st.session_state.dense_weight = st.slider(
            "Semantic ← → Keyword balance:",
            0.0, 1.0, st.session_state.get("dense_weight", DENSE_WEIGHT), 0.05,
            help="1.0 uses semantic similarity only; 0.0 uses keyword matching only. "
                 "0.5 weights both equally. Lower it when exact terms matter more "
                 "than meaning (part numbers, gene names); raise it for paraphrased "
                 "or conceptual questions."
        )
        st.caption(
            f"semantic {st.session_state.dense_weight:.0%} · "
            f"keyword {1 - st.session_state.dense_weight:.0%}"
        )

        st.session_state.final_k = st.slider(
            "Passages sent to the model:",
            1, 15, st.session_state.get("final_k", FINAL_TOP_K), 1,
            help="More passages means more evidence but a fuller context window."
        )

        st.session_state.fused_k = st.slider(
            "Candidates considered:",
            st.session_state.get("final_k", FINAL_TOP_K), 40,
            max(st.session_state.get("fused_k", FUSED_TOP_K),
                st.session_state.get("final_k", FINAL_TOP_K)), 1,
            help="Retrieved and ranked before the best are selected. Raising this "
                 "helps most when re-ranking is enabled."
        )

        st.session_state.dense_k = st.slider(
            "Candidates per index:",
            5, 50, st.session_state.get("dense_k", DENSE_TOP_K), 5,
            help="How deep each of the two indexes searches before their results "
                 "are fused."
        )
        st.session_state.sparse_k = st.session_state.dense_k

        # Metadata pre-filtering: restrict retrieval before it runs, rather
        # than filtering results afterwards.
        st.markdown("**Restrict search to:**")
        st.session_state.filter_files = st.multiselect(
            "Documents:",
            _store.source_files,
            default=st.session_state.get("filter_files", []),
            help="Leave empty to search everything.",
            label_visibility="collapsed"
        )

        st.session_state.generate_followups = st.checkbox(
            "Generate follow-up questions",
            value=st.session_state.get("generate_followups", True),
            help="Costs one extra model call per answer. Turn off for faster replies."
        )

    # -----------------------------
    # Sidebar: Advanced Techniques
    # -----------------------------
    with st.sidebar.expander("🧪 Advanced Techniques", expanded=False):
        st.caption(
            "Each option below costs one extra model call per question. "
            "On local hardware that is seconds, not milliseconds — turn on what "
            "earns its keep for your documents."
        )

        st.markdown("**Query transformation**")
        st.session_state.condense_history = st.checkbox(
            "Resolve follow-up references",
            value=st.session_state.get("condense_history", True),
            help="Rewrites 'how does that compare?' into a standalone question using "
                 "the conversation. Without it, follow-up questions retrieve poorly, "
                 "because pronouns carry no searchable meaning. Only runs when the "
                 "question actually looks like a follow-up."
        )
        st.session_state.rewrite_query = st.checkbox(
            "Rewrite query for retrieval",
            value=st.session_state.get("rewrite_query", False),
            help="Restates your question in the terminology the documents use "
                 "before searching."
        )
        st.session_state.multi_query = st.checkbox(
            "Search multiple phrasings",
            value=st.session_state.get("multi_query", False),
            help="Generates 3 alternative phrasings, searches each, and fuses the "
                 "results. Best single improvement to recall."
        )
        st.session_state.use_hyde = st.checkbox(
            "HyDE (search by hypothetical answer)",
            value=st.session_state.get("use_hyde", False),
            help="Writes a plausible answer and searches with that instead of the "
                 "question, since answers resemble documents more than questions do. "
                 "Affects semantic search only; keyword search still uses your words."
        )
        st.session_state.infer_filter = st.checkbox(
            "Infer document/section filter",
            value=st.session_state.get("infer_filter", False),
            help="Detects when you name a document or section ('in the methods "
                 "section of the Smith paper') and restricts the search accordingly."
        )

        st.markdown("**Re-ranking**")
        st.session_state.use_rerank = st.checkbox(
            "Re-rank retrieved passages",
            value=st.session_state.get("use_rerank", False),
            help="Retrieves more candidates than needed, then re-scores them "
                 "against your question and keeps the best. The largest precision "
                 "gain available here."
        )
        if st.session_state.use_rerank:
            st.session_state.rerank_backend = st.radio(
                "Re-ranker:",
                ["llm", "cross-encoder"],
                index=0 if st.session_state.get("rerank_backend", "llm") == "llm" else 1,
                format_func=lambda b: (
                    "Local LLM (no extra install)" if b == "llm"
                    else "Cross-encoder (needs sentence-transformers)"
                ),
                help="The LLM re-ranker scores every candidate in one call. The "
                     "cross-encoder is more accurate and faster per query, but "
                     "requires installing sentence-transformers."
            )

# Initialize LLM engine if model is selected
llm_engine = None
if selected_model:
    llm_engine = ChatOllama(
        model=selected_model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_ctx=num_ctx
    )

# -----------------------------
# Sidebar: Question History
# -----------------------------
if st.session_state.chat_history:
    st.sidebar.title("❓ Questions Asked")
    for i, (query, _) in enumerate(st.session_state.chat_history):
        # Truncate long queries for display
        display_query = query[:50] + "..." if len(query) > 50 else query
        st.sidebar.markdown(f"[**{i+1}.** {display_query}](#question-{i})")
        st.sidebar.markdown("---")

# =============================================================================
# PAGE 1: CHAT INTERFACE
# =============================================================================
if page == "Chat":
    st.title("🔬 Local LLM Scientific Assistant")
    
    # Show current mode with clear visual indicators
    _chunk_count = st.session_state.rag_store.count if index_is_populated() else 0

    if _chunk_count:
        if st.session_state.allow_llm_knowledge:
            st.warning("⚠️ **HYBRID MODE**: AI will use documents + general knowledge")
            st.caption(f"🎯 **Perspective:** {st.session_state.perspective} | **Audience:** {st.session_state.audience} | **Documents:** {len(st.session_state.uploaded_documents)} | **Indexed chunks:** {_chunk_count:,} | **Mode:** Hybrid")
        else:
            st.success("✅ **STRICT RAG MODE**: AI will answer ONLY from your uploaded documents")
            st.caption(f"🎯 **Perspective:** {st.session_state.perspective} | **Audience:** {st.session_state.audience} | **Documents:** {len(st.session_state.uploaded_documents)} | **Indexed chunks:** {_chunk_count:,} | **Mode:** RAG Only")
    elif st.session_state.document_content:
        st.warning(
            "⚠️ **DOCUMENTS NOT INDEXED**: text was extracted but retrieval is "
            "unavailable, so answers come from general knowledge. Check that an "
            "embedding model is installed (`ollama pull nomic-embed-text`)."
        )
        st.caption(f"🎯 **Perspective:** {st.session_state.perspective} | **Audience:** {st.session_state.audience}")
    else:
        st.info("💬 **LLM MODE**: AI will use general knowledge (upload documents for RAG mode)")
        st.caption(f"🎯 **Perspective:** {st.session_state.perspective} | **Audience:** {st.session_state.audience}")
    
    # -----------------------------
    # Document Upload Section
    # -----------------------------
    with st.expander("📄 Upload Documents (PDF, DOCX, PPTX)", expanded=False):
        st.markdown("*Upload documents to include their content in your queries*")

        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF, DOCX, or PPTX files",
            type=['pdf', 'docx', 'doc', 'pptx'],
            accept_multiple_files=True,
            help="Upload one or more documents. Maximum 200 MB per file.",
            key="document_uploader"
        )
        
        # Ingest-time enrichment. Placed here rather than in the sidebar so its
        # cost is visible at the moment the user commits to paying it.
        st.session_state.enrich_chunks = st.checkbox(
            "🧠 Add contextual headers (slow, improves retrieval)",
            value=st.session_state.get("enrich_chunks", False),
            help="Writes a sentence of surrounding context into each chunk before "
                 "embedding, so passages full of vague references ('the value rose "
                 "to 12%') stay findable. Costs one model call per chunk — minutes, "
                 "not seconds, on a large document. The chunk text you see cited is "
                 "unchanged; only its vector is enriched."
        )
        if st.session_state.enrich_chunks:
            st.caption(
                "⏳ A 100-page document is roughly 150 chunks — about 150 extra "
                "model calls. Since the index is cleared when the app closes, "
                "this cost is paid again each session."
            )

        # Process uploaded files
        if uploaded_files:
            if st.button("📥 Process Documents", type="primary"):
                with st.spinner("Extracting text..."):
                    results = DocumentProcessor.process_multiple_documents(uploaded_files)
                    st.session_state.uploaded_documents = results

                    # Kept for preview and for the no-retrieval fallback path
                    st.session_state.document_content = DocumentProcessor.format_document_content(
                        results,
                        include_filenames=True
                    )

                    # Reset to STRICT RAG MODE when documents are loaded
                    st.session_state.allow_llm_knowledge = False

                # Index for retrieval (outside the spinner: this shows its own
                # progress bar and is the slow part on large uploads)
                indexed = 0
                index_error = None
                if get_store() is not None:
                    try:
                        indexed = ingest_documents(results, llm_engine)
                        st.session_state.index_stale_reason = None
                    except EmbeddingError as e:
                        index_error = str(e)

                for result in results:
                    summary = DocumentProcessor.get_document_summary(result)
                    if result['error']:
                        st.error(summary)
                    else:
                        st.info(summary)
                    if result.get('warning'):
                        st.warning(f"⚠️ {result['filename']}: {result['warning']}")

                if index_error:
                    st.error(f"❌ Indexing failed: {index_error}")
                elif indexed:
                    st.success(f"✅ Indexed {indexed:,} chunks — documents are now searchable.")
                elif get_store() is None:
                    st.warning(
                        "⚠️ No embedding model available, so documents were extracted but "
                        "not indexed. Install one with `ollama pull nomic-embed-text`."
                    )

                st.rerun()

        # Display currently loaded documents
        if st.session_state.uploaded_documents:
            st.markdown("---")
            st.markdown("**Currently Loaded Documents:**")

            for i, result in enumerate(st.session_state.uploaded_documents):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if result['error']:
                        st.markdown(f"❌ {result['filename']} - *Error*")
                    else:
                        word_count = len(result['content'].split())
                        st.markdown(f"✅ {result['filename']} (~{word_count:,} words)")
                with col2:
                    if st.button("🗑️", key=f"remove_doc_{i}", help="Remove this document"):
                        st.session_state.uploaded_documents.pop(i)
                        st.session_state.document_content = DocumentProcessor.format_document_content(
                            st.session_state.uploaded_documents,
                            include_filenames=True
                        )
                        # The index must follow the document list, or the
                        # removed file keeps answering questions.
                        rebuild_index(llm_engine)
                        st.rerun()

            # Clear all button
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.session_state.uploaded_documents = []
                st.session_state.document_content = ""
                store = get_store()
                if store is not None:
                    store.clear()
                st.rerun()
            
            # Preview toggle
            if st.checkbox("👁️ Preview Document Content", key="preview_docs"):
                st.markdown("**Document Content Preview:**")
                preview_text = st.session_state.document_content[:2000]
                if len(st.session_state.document_content) > 2000:
                    preview_text += f"\n\n... ({len(st.session_state.document_content) - 2000:,} more characters)"
                st.text_area(
                    "Content",
                    preview_text,
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )

    # -----------------------------
    # Chunk Inspector
    # -----------------------------
    _inspect_store = get_store()
    if _inspect_store is not None and not _inspect_store.is_empty:
        with st.expander("🔍 Inspect Indexed Chunks", expanded=False):
            st.markdown(
                "*What retrieval actually searches over. Useful for sanity-checking "
                "how a document was split and whether page/section metadata survived.*"
            )

            _stats = _inspect_store.stats()
            col1, col2, col3 = st.columns(3)
            col1.metric("Chunks", f"{_stats['chunks']:,}")
            col2.metric("Documents", _stats['documents'])
            col3.metric("Vector dim", _stats['dimension'] or "—")

            file_choice = st.selectbox(
                "Filter by document:",
                ["All"] + _inspect_store.source_files,
                key="chunk_inspector_file"
            )

            visible = _inspect_store.all_chunks(
                None if file_choice == "All" else file_choice
            )

            st.caption(f"Showing {min(len(visible), 25)} of {len(visible):,} chunks")

            for chunk in visible[:25]:
                st.markdown(f"**`{chunk.chunk_id}`** — {chunk.location_label}")
                st.caption(
                    f"{len(chunk.text):,} chars · {len(chunk.text.split()):,} words · "
                    f"parent `{chunk.parent_id}` ({len(chunk.parent_text):,} chars)"
                )
                st.text(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))
                st.markdown("---")

    # -----------------------------
    # Question Templates Section
    # -----------------------------
    with st.expander("📋 Question Templates", expanded=False):
        st.markdown("*Select a pre-defined question template or ask your own*")
        
        # Load templates and organize by category
        all_templates = template_manager.load_templates()
        categories = template_manager.get_all_categories()
        
        # Category filter
        selected_category = st.selectbox(
            "Filter by Category:",
            ["All"] + categories,
            key="category_filter"
        )
        
        # Filter templates by category
        if selected_category == "All":
            filtered_templates = all_templates
        else:
            filtered_templates = template_manager.get_templates_by_category(selected_category)
        
        if filtered_templates:
            # Create template selector
            template_names = [t["name"] for t in filtered_templates]
            selected_template_name = st.selectbox(
                "Select Template:",
                ["-- None --"] + template_names,
                key="template_selector"
            )
            
            if selected_template_name != "-- None --":
                selected_template = template_manager.get_template_by_name(selected_template_name)
                
                if selected_template:
                    # Display template details
                    st.info(f"**Description:** {selected_template.get('description', 'No description available')}")
                    st.markdown(f"**Template:** {selected_template['template']}")
                    
                    # Two options for using the template
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📝 Use Template with Form", use_container_width=True):
                            st.session_state.selected_template = selected_template
                            st.rerun()
                    
                    with col2:
                        if st.button("💬 Populate Chat Input", use_container_width=True):
                            # This will populate the chat input below
                            st.session_state.template_text = selected_template['template']
                            st.success("✅ Template populated in chat input below!")
        else:
            st.info("No templates available in this category.")
    
    # -----------------------------
    # Template Input Form (if template selected)
    # -----------------------------
    if st.session_state.selected_template:
        with st.form("template_form"):
            st.subheader(f"📋 {st.session_state.selected_template['name']}")
            st.markdown(f"*{st.session_state.selected_template.get('description', '')}*")
            
            # Show the template text
            st.markdown(f"**Question:** {st.session_state.selected_template['template']}")
            
            # Get user input if required
            user_content = ""
            if st.session_state.selected_template.get('requires_input', True):
                user_content = st.text_area(
                    st.session_state.selected_template.get('input_label', 'Enter your content:'),
                    height=150,
                    key="template_content_input"
                )
            
            col1, col2 = st.columns([1, 1])
            submit = col1.form_submit_button("🚀 Submit Question", use_container_width=True)
            cancel = col2.form_submit_button("❌ Cancel", use_container_width=True)
            
            if submit:
                if st.session_state.selected_template.get('requires_input', True) and not user_content.strip():
                    st.error("Please provide the required input.")
                else:
                    # Combine template and content
                    if user_content.strip():
                        base_query = f"{st.session_state.selected_template['template']}\n\n{user_content}"
                    else:
                        base_query = st.session_state.selected_template['template']
                    
                    if not llm_engine:
                        st.error("No model selected. Please select a model in the sidebar.")
                    else:
                        with st.spinner("🤔 Retrieving and generating response..."):
                            run_query(base_query, llm_engine, num_ctx)
                        st.session_state.selected_template = None
                        st.rerun()
            
            if cancel:
                st.session_state.selected_template = None
                st.rerun()
    
    # -----------------------------
    # Chat Input
    # -----------------------------
    # Check if we have template text to populate
    initial_input = st.session_state.get('template_text', '')
    if initial_input:
        # Clear it after reading
        del st.session_state.template_text
    
    user_query = st.chat_input(
        "Type your query here or use a template above...",
        key="main_chat_input"
    )

    # If a follow-up question button was clicked, treat it as the user query
    if not user_query and st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_query:
        if llm_engine:
            with st.spinner("🤔 Retrieving and generating response..."):
                run_query(user_query, llm_engine, num_ctx)
            st.rerun()
        else:
            st.error("⚠️ No model selected. Please select a model in the sidebar.")
    
    # -----------------------------
    # Display Conversation History
    # -----------------------------
    st.markdown("---")
    st.header("💬 Conversation History")
    
    if st.session_state.chat_history:
        # Full conversation export button
        full_html = build_conversation_html(
            st.session_state.chat_history,
            title="Conversation Transcript",
            perspective=st.session_state.perspective,
            audience=st.session_state.audience
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.download_button(
                label="⬇️ Download Full Conversation (HTML)",
                data=full_html.encode("utf-8"),
                file_name=safe_filename("conversation_transcript"),
                mime="text/html",
                use_container_width=True
            )
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        # Display each Q&A pair
        for i, (query, response) in enumerate(st.session_state.chat_history):
            # Create anchor for sidebar navigation
            st.markdown(f"<a name='question-{i}'></a>", unsafe_allow_html=True)
            
            # User question
            with st.container():
                st.markdown(f"### 👤 You (Question #{i+1})")
                st.markdown(query)
            
            # AI response
            with st.container():
                st.markdown(f"### 🤖 AI Response")
                st.markdown(response)

            # Citation check: fabricated markers are surfaced immediately,
            # since they are indistinguishable from real ones in the prose.
            citations = st.session_state.answer_citations.get(i)
            if citations is not None and not citations.is_clean:
                st.warning(citations.summary())

            # Sources actually used to ground this answer
            sources = st.session_state.answer_sources.get(i, [])
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                    if citations is not None:
                        st.caption(citations.summary())
                    for source in sources:
                        st.markdown(f"**{source.marker}** {source.location_label}")
                        score_bits = [f"retrieved by: {source.retrieval_source}"]
                        if source.fused_score is not None:
                            score_bits.append(f"fused: {source.fused_score:.5f}")
                        if source.rerank_score is not None:
                            score_bits.append(f"rerank: {source.rerank_score:.2f}")
                        st.caption(" · ".join(score_bits))
                        st.text(
                            source.text[:600] + ("..." if len(source.text) > 600 else "")
                        )
                        st.markdown("---")

            # Stage-by-stage trace: makes it possible to tell a retrieval
            # failure from a generation failure, which is otherwise guesswork.
            diagnostics = st.session_state.answer_diagnostics.get(i)
            if diagnostics is not None:
                with st.expander("🔬 Retrieval Diagnostics", expanded=False):
                    if diagnostics.search_query != diagnostics.original_query:
                        st.markdown(f"**Original query:** {diagnostics.original_query}")
                        st.markdown(f"**Search query:** {diagnostics.search_query}")
                    if diagnostics.condensed_query:
                        st.markdown(f"**Condensed:** {diagnostics.condensed_query}")
                    if diagnostics.rewritten_query:
                        st.markdown(f"**Rewritten:** {diagnostics.rewritten_query}")
                    if diagnostics.query_variants:
                        st.markdown(
                            "**Variants:** " + " · ".join(diagnostics.query_variants)
                        )
                    if diagnostics.where_filter:
                        st.markdown(f"**Metadata filter:** `{diagnostics.where_filter}`")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Semantic hits", diagnostics.dense_hits)
                    col2.metric("Keyword hits", diagnostics.sparse_hits)
                    col3.metric("Sources used", diagnostics.sources_used)

                    if diagnostics.context_budget:
                        pct = 100.0 * diagnostics.context_chars / diagnostics.context_budget
                        st.caption(
                            f"Context: {diagnostics.context_chars:,} of "
                            f"{diagnostics.context_budget:,} chars ({pct:.0f}% of budget)"
                            + (" · re-ranked" if diagnostics.reranked else "")
                        )
                    for note in diagnostics.notes:
                        st.info(note)

            # Suggested follow-up questions
            followups = st.session_state.followup_questions.get(i, [])
            if followups:
                st.markdown("**💡 Suggested Follow-up Questions:**")
                for j, fq in enumerate(followups):
                    if st.button(f"❓ {fq}", key=f"followup-{i}-{j}"):
                        st.session_state.pending_query = fq
                        st.rerun()

            # Download individual Q&A
            pair_html = build_pair_html(
                i, query, response,
                perspective=st.session_state.perspective,
                audience=st.session_state.audience
            )
            st.download_button(
                label=f"⬇️ Download Q&A #{i+1} (HTML)",
                data=pair_html.encode("utf-8"),
                file_name=safe_filename(f"qa_{i+1}"),
                mime="text/html",
                key=f"dl-qapair-{i}",
                use_container_width=False
            )
            
            st.markdown("---")
    else:
        st.info("👋 No conversation yet. Ask a question to get started!")

# =============================================================================
# PAGE 2: TEMPLATE MANAGEMENT
# =============================================================================
elif page == "Manage Templates":
    st.title("📚 Template Management")
    st.markdown("*Create, edit, and organize your question templates*")
    
    # Create tabs for different template operations
    tab1, tab2, tab3 = st.tabs(["📋 View Templates", "➕ Add Template", "🔧 Settings"])
    
    # -----------------------------
    # TAB 1: View/Delete Templates
    # -----------------------------
    with tab1:
        st.header("All Templates")
        
        all_templates = template_manager.load_templates()
        categories = template_manager.get_all_categories()
        
        # Category filter
        view_category = st.selectbox(
            "Filter by Category:",
            ["All"] + categories,
            key="view_category_filter"
        )
        
        # Filter templates
        if view_category == "All":
            display_templates = all_templates
        else:
            display_templates = template_manager.get_templates_by_category(view_category)
        
        if display_templates:
            st.markdown(f"**Total Templates:** {len(display_templates)}")
            
            # Display templates in expandable sections
            for template in display_templates:
                with st.expander(f"📌 {template['name']} ({template.get('category', 'Uncategorized')})"):
                    st.markdown(f"**ID:** `{template['id']}`")
                    st.markdown(f"**Category:** {template.get('category', 'Uncategorized')}")
                    st.markdown(f"**Description:** {template.get('description', 'No description')}")
                    st.markdown(f"**Template Text:**")
                    st.code(template['template'], language=None)
                    st.markdown(f"**Requires Input:** {'Yes' if template.get('requires_input', True) else 'No'}")
                    
                    if template.get('requires_input', True):
                        st.markdown(f"**Input Label:** {template.get('input_label', 'N/A')}")
                    
                    # Delete button
                    if st.button(f"🗑️ Delete Template", key=f"delete_{template['id']}"):
                        if template_manager.delete_template(template['id']):
                            st.success(f"✅ Deleted template: {template['name']}")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete template")
        else:
            st.info("No templates found in this category.")
    
    # -----------------------------
    # TAB 2: Add New Template
    # -----------------------------
    with tab2:
        st.header("Create New Template")
        
        with st.form("add_template_form"):
            new_name = st.text_input(
                "Template Name*:",
                help="Give your template a descriptive name"
            )
            
            new_category = st.selectbox(
                "Category*:",
                ["Research", "Teaching", "Laboratory", "Research Design", 
                 "Statistics", "Critical Thinking", "Career", "General", "Custom"],
                help="Organize templates by category"
            )
            
            new_template = st.text_area(
                "Template Text*:",
                height=150,
                help="The question or prompt text. Use clear, specific language."
            )
            
            new_description = st.text_area(
                "Description:",
                height=80,
                help="Brief description of what this template does (optional but recommended)"
            )
            
            new_requires_input = st.checkbox(
                "Requires user input",
                value=True,
                help="Check if this template needs additional content from the user"
            )
            
            new_input_label = ""
            if new_requires_input:
                new_input_label = st.text_input(
                    "Input Label:",
                    value="Enter your content:",
                    help="Label for the input field when using this template"
                )
            
            submitted = st.form_submit_button("➕ Add Template", use_container_width=True)
            
            if submitted:
                if not new_name.strip():
                    st.error("❌ Template name is required")
                elif not new_template.strip():
                    st.error("❌ Template text is required")
                else:
                    success = template_manager.add_template(
                        name=new_name.strip(),
                        template=new_template.strip(),
                        category=new_category,
                        requires_input=new_requires_input,
                        input_label=new_input_label,
                        description=new_description.strip()
                    )
                    
                    if success:
                        st.success(f"✅ Successfully added template: {new_name}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Failed to add template")
    
    # -----------------------------
    # TAB 3: Settings
    # -----------------------------
    with tab3:
        st.header("Template Settings")
        
        st.subheader("📊 Statistics")
        all_templates = template_manager.load_templates()
        categories = template_manager.get_all_categories()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Templates", len(all_templates))
        with col2:
            st.metric("Categories", len(categories))
        
        st.markdown("---")
        
        st.subheader("⚠️ Danger Zone")
        st.warning("**Warning:** The following action cannot be undone!")
        
        if st.button("🔄 Reset to Default Templates", type="secondary"):
            st.session_state.confirm_reset = True
        
        if st.session_state.get('confirm_reset', False):
            st.error("**Are you sure?** This will delete all custom templates and restore defaults.")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Yes, Reset", type="primary"):
                    if template_manager.reset_to_defaults():
                        st.success("✅ Templates reset to defaults")
                        st.session_state.confirm_reset = False
                        st.rerun()
                    else:
                        st.error("❌ Failed to reset templates")
            
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_reset = False
                    st.rerun()

# -----------------------------
# Footer
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p>🔬 Local LLM Scientific Assistant</p>
    <p>Powered by Ollama & Streamlit</p>
</div>
""", unsafe_allow_html=True)
