"""
Answering questions against a persistent library.

The same flow as `app.run_query` in the original app -- grounding mode, query
transformation, hybrid retrieval, optional re-ranking, citation checking,
follow-up suggestions -- written once as a UI-free, testable module. LLM access
is injected:

  * chat_llm(messages) -> str, where messages is a list of (role, content) with
    role in {"system", "user", "assistant"};
  * ask_llm(prompt) -> str for single-prompt helper calls.

This is the one `rag_core` module that imports from `utils/`, for the shared
`PromptBuilder`, so both apps word their grounding rules identically.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from utils.prompt_builder import PromptBuilder

from .context_assembler import Source
from .conversations import HISTORY_WINDOW, PASSAGE_CHARS, SavedSource
from .library import ChatbotSettings, Library
from .pipeline import MODE_STRICT, RagResult, answer as rag_answer, resolve_mode
from .query_transformer import QueryTransformer
from .reranker import build_reranker

ChatLLM = Callable[[List[Tuple[str, str]]], str]
AskLLM = Callable[[str], str]


@dataclass
class AskResult:
    rag: RagResult
    followups: List[str] = field(default_factory=list)


def build_system_prompt(settings: ChatbotSettings, has_documents: bool, grounding_mode: str) -> str:
    """
    The app's standard system prompt, with the library's custom instructions
    appended. Appending (never replacing) keeps the grounding and citation
    rules in force whatever the instructions say.
    """
    prompt = PromptBuilder.build_system_prompt(
        settings.perspective, settings.audience, has_documents, grounding_mode
    )
    custom = (settings.custom_instructions or "").strip()
    if custom:
        prompt += (
            "\n\nAdditional instructions for this chatbot (these add to, and never "
            "override, the rules above):\n" + custom
        )
    return prompt


def generate_followups(question: str, answer: str, ask_llm: AskLLM) -> List[str]:
    """Three follow-up questions; same prompt and parsing as the original app."""
    prompt = (
        "Based on the following question and answer, generate exactly 3 insightful "
        "follow-up questions that would deepen understanding of the topic. "
        "Return only the 3 questions, one per line, numbered 1. 2. 3. "
        "Do not include any other text or explanation.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Follow-up questions:"
    )
    try:
        raw = ask_llm(prompt) or ""
    except Exception:
        return []
    questions = []
    for line in raw.strip().split("\n"):
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip()).strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:3]


def collection_filter(settings: ChatbotSettings, all_collections: Sequence[str]) -> Optional[Dict]:
    """Chroma `where` for the chosen collections; None means search everything."""
    chosen = [c for c in settings.collections if c in all_collections]
    if not chosen or set(chosen) >= set(all_collections):
        return None
    return {"collection": {"$in": chosen}}


def saved_sources(library: Library, sources: Sequence[Source]) -> List[SavedSource]:
    """Sources as stored with a conversation: copied passage + owning document."""
    out = []
    for source in sources:
        chunk = library.store.get_chunk(source.chunk_id)
        doc_id = str(chunk.extra.get("doc_id", "")) if chunk else ""
        out.append(SavedSource(
            marker=source.marker,
            doc_id=doc_id,
            location_label=source.location_label,
            passage=(source.text or "")[:PASSAGE_CHARS],
        ))
    return out


def ask(
    library: Library,
    question: str,
    history: Sequence[Tuple[str, str]],
    chat_llm: ChatLLM,
    ask_llm: AskLLM,
) -> AskResult:
    """Answer one question with the library's saved chatbot settings."""
    settings = library.info.chatbot
    store = library.store
    has_index = not store.is_empty
    # Unlike the original app, an empty library does not quietly switch to
    # general knowledge: "Allow LLM general knowledge" off means documents only.
    mode = resolve_mode(True, settings.allow_llm_knowledge)
    recent = list(history)[-HISTORY_WINDOW:]

    def generate(user_message: str, grounding_mode: str, has_context: bool) -> str:
        messages: List[Tuple[str, str]] = [
            ("system", build_system_prompt(settings, has_context, grounding_mode))
        ]
        for past_question, past_answer in recent:
            messages.append(("user", past_question))
            messages.append(("assistant", past_answer))
        messages.append(("user", user_message))
        return chat_llm(messages)

    retrieval = settings.retrieval_settings()

    transformer = None
    if has_index and any([retrieval.condense_history, retrieval.rewrite_query,
                          retrieval.multi_query, retrieval.hyde, retrieval.infer_filter]):
        sections = sorted({c.section for c in store.all_chunks() if c.section}) \
            if retrieval.infer_filter else []
        transformer = QueryTransformer(
            ask_llm, known_files=store.source_files, known_sections=sections
        )

    reranker = None
    rerank_note = None
    if has_index and retrieval.rerank:
        try:
            reranker = build_reranker(settings.rerank_backend, llm=ask_llm)
        except ImportError as e:
            rerank_note = f"Re-ranking skipped: {e}"

    result = rag_answer(
        question=question,
        generate=generate,
        store=store if has_index else None,
        mode=mode,
        num_ctx=settings.num_ctx,
        settings=retrieval,
        where=collection_filter(settings, library.collections()) if has_index else None,
        transformer=transformer,
        reranker=reranker,
        chat_history=recent,
    )
    if rerank_note:
        result.diagnostics.notes.append(rerank_note)
    if mode == MODE_STRICT and not has_index:
        result.answer = (
            "No documents are indexed in this library yet, so there is nothing to "
            "answer from. Add documents on the Documents tab, or turn on "
            "'Allow LLM general knowledge' in Chatbot settings."
        )

    # Follow-ups only when the model actually answered, not for a refusal.
    answered = bool(result.sources) or result.mode != MODE_STRICT
    followups: List[str] = []
    if settings.generate_followups and answered:
        followups = generate_followups(question, result.answer, ask_llm)
    return AskResult(rag=result, followups=followups)
