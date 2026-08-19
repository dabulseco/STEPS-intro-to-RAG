"""
Query transformation.

The text a person types is tuned for a human reader, not for a retrieval index.
"How does that compare to the control?" carries almost no retrievable signal:
the subject lives in the previous turn, and the words that remain are mostly
grammar. This module rewrites the question before it reaches the index.

Five transformations, each independently switchable:

  * **condensation** -- resolve pronouns against the conversation so a
    follow-up becomes a standalone question;
  * **rewriting** -- restate the question in the vocabulary documents use;
  * **multi-query** -- fan out into several phrasings and fuse the results;
  * **HyDE** -- embed a hypothetical *answer* rather than the question, so the
    query vector lands where real answers live;
  * **filter inference** -- turn "in the Smith paper's methods" into a metadata
    pre-filter applied before the search runs.

Every transformation degrades safely: if the model returns something unusable,
the original question is kept. A bad rewrite is worse than no rewrite, so the
sanitizer here is deliberately suspicious of its own model's output.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Words whose referent lives outside the sentence. Their presence is what makes
# a question unanswerable on its own, and so worth spending a call to condense.
_DEPENDENT_MARKERS = {
    "it", "its", "that", "this", "these", "those", "they", "them", "their",
    "he", "she", "his", "her", "him", "the former", "the latter", "there",
    "same", "such", "above", "previous", "earlier", "again", "instead",
}

# Words that mark a lead-in phrase rather than content. Small local models
# habitually prepend one despite being told not to.
_PREAMBLE_KEYWORDS = (
    "sure", "certainly", "okay", "here", "rewritten", "standalone", "search",
    "query", "queries", "answer", "output", "result", "response", "following",
)

# A head starting with one of these is the question itself, not a lead-in, so
# a colon after it must not be treated as the end of a preamble.
_QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "which", "does", "did",
    "is", "are", "was", "were", "can", "could", "should", "would", "list",
    "describe", "explain", "summarize", "compare",
)


def _strip_preamble(line: str) -> str:
    """Remove a leading 'Sure, here is the query:' style lead-in."""
    if ":" not in line[:80]:
        return line

    head, _, tail = line.partition(":")
    head_lower = head.strip().lower()

    if len(head) > 60:
        return line
    if head_lower.startswith(_QUESTION_STARTERS):
        return line
    if "?" in head:
        return line
    if not any(keyword in head_lower for keyword in _PREAMBLE_KEYWORDS):
        return line

    return tail.strip()


@dataclass
class TransformedQuery:
    """The result of transforming one question."""

    search_query: str
    variants: List[str] = field(default_factory=list)
    condensed_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    hyde_text: Optional[str] = None
    where_filter: Optional[Dict] = None
    notes: List[str] = field(default_factory=list)


def _clean_single_line(raw: str, max_chars: int = 400) -> Optional[str]:
    """
    Coerce a model response into one usable query line, or None.

    Returning None rather than a best-effort guess is the point: the caller
    falls back to the user's own words, which are never worse than a mangled
    rewrite.
    """
    if not raw:
        return None

    text = raw.strip()

    # Reasoning-style models sometimes emit a think block before the answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Walk the lines rather than taking the first: when a model ignores the
    # "no preamble" instruction it usually puts the lead-in on its own line and
    # the actual query on the next one.
    for line in (ln.strip() for ln in text.split("\n")):
        if not line:
            continue

        candidate = _strip_preamble(line)

        # Strip list numbering ("1. ", "- ") left over from enumerated output.
        candidate = re.sub(r"^[\d]+[\.\)]\s*", "", candidate)
        candidate = re.sub(r"^[-*]\s*", "", candidate)
        candidate = candidate.strip('"\'` ').strip()

        if not candidate or len(candidate) > max_chars:
            continue

        # Still a sentence *about* the task rather than a query.
        if candidate.lower().startswith(("i cannot", "i can't", "as an ai", "sorry")):
            return None

        return candidate

    return None


def looks_dependent(question: str) -> bool:
    """
    Whether a question appears to depend on earlier conversation.

    Used to skip condensation when it cannot help. On local hardware every
    avoided model call is a second or two off the response time, and a
    self-contained question gains nothing from being rewritten.
    """
    words = re.findall(r"[a-z']+", question.lower())
    if not words:
        return False

    # Very short questions are usually elliptical ("why?", "and the control?").
    if len(words) <= 4:
        return True

    return any(w in _DEPENDENT_MARKERS for w in words)


class QueryTransformer:
    """
    LLM-backed query rewriting.

    Args:
        llm: Callable taking a prompt and returning the model's text. Supplied
            by the caller so this module stays free of any LLM library.
        known_files: Filenames available for metadata filtering
        known_sections: Section names available for metadata filtering
    """

    def __init__(
        self,
        llm: Callable[[str], str],
        known_files: Optional[Sequence[str]] = None,
        known_sections: Optional[Sequence[str]] = None,
    ):
        self.llm = llm
        self.known_files = list(known_files or [])
        self.known_sections = list(known_sections or [])

    # -------------------------------------------------------------------------
    # Individual transformations
    # -------------------------------------------------------------------------
    def condense(self, question: str, chat_history: Sequence[Tuple[str, str]]) -> Optional[str]:
        """Rewrite a follow-up into a standalone question using recent turns."""
        if not chat_history or not looks_dependent(question):
            return None

        recent = chat_history[-3:]
        transcript = "\n".join(
            f"User: {q}\nAssistant: {a[:300]}" for q, a in recent
        )

        prompt = (
            "Rewrite the follow-up question as a standalone question that can be "
            "understood without the conversation. Replace pronouns and references "
            "with the specific things they refer to. Keep it short. Output only the "
            "rewritten question on a single line, with no preamble.\n\n"
            f"Conversation:\n{transcript}\n\n"
            f"Follow-up question: {question}\n\n"
            "Standalone question:"
        )

        try:
            return _clean_single_line(self.llm(prompt))
        except Exception:
            return None

    def rewrite(self, question: str) -> Optional[str]:
        """Restate a question in the vocabulary a document would use."""
        prompt = (
            "Rewrite the question as a search query for finding relevant passages "
            "in a technical document collection. Use the formal terminology a "
            "document would use, keep every specific name, number and identifier "
            "from the original, and drop conversational filler. Output only the "
            "query on a single line, with no preamble.\n\n"
            f"Question: {question}\n\n"
            "Search query:"
        )

        try:
            rewritten = _clean_single_line(self.llm(prompt))
        except Exception:
            return None

        # A rewrite that dropped almost everything usually means the model
        # answered the question instead of restating it.
        if rewritten and len(rewritten) < max(8, len(question) // 5):
            return None
        return rewritten

    def hyde(self, question: str, max_chars: int = 800) -> Optional[str]:
        """
        Write a hypothetical answer to embed in place of the question.

        A question and the passage that answers it are written differently --
        one asks, the other states -- so their embeddings sit further apart than
        the topic warrants. Embedding an invented answer instead puts the query
        vector in the same neighbourhood as real answers. The passage does not
        need to be factually right; only its shape and vocabulary matter, and it
        is never shown to the user or used as evidence.
        """
        prompt = (
            "Write a short passage (2-3 sentences) that plausibly answers the "
            "question, as it might appear in a technical document. Use the "
            "terminology such a document would use. Do not hedge, do not mention "
            "being an AI, and output only the passage.\n\n"
            f"Question: {question}\n\n"
            "Passage:"
        )

        try:
            raw = self.llm(prompt) or ""
        except Exception:
            return None

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)

        lines = [_strip_preamble(ln.strip()) for ln in raw.split("\n")]
        passage = " ".join(ln for ln in lines if ln).strip().strip('"\'` ')

        if not passage or len(passage) < 20:
            return None
        return passage[:max_chars]

    def multi_query(self, question: str, n: int = 3) -> List[str]:
        """Generate alternative phrasings to retrieve against."""
        prompt = (
            f"Write {n} different search queries that would find passages answering "
            "the question below. Vary the wording and the level of specificity; use "
            "synonyms a document might use. Output one query per line, numbered "
            f"1. to {n}. No other text.\n\n"
            f"Question: {question}\n\n"
            "Queries:"
        )

        try:
            raw = self.llm(prompt)
        except Exception:
            return []

        raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL | re.IGNORECASE)

        variants = []
        for line in raw.split("\n"):
            cleaned = _clean_single_line(line)
            if cleaned and cleaned.lower() != question.lower() and cleaned not in variants:
                variants.append(cleaned)

        return variants[:n]

    def infer_filter(self, question: str) -> Optional[Dict]:
        """
        Derive a Chroma metadata filter from the question, if one is implied.

        Only values that actually exist in the index are accepted -- a filter
        naming a file that is not there would match nothing and silently return
        an empty answer, which is far worse than not filtering at all.
        """
        if not self.known_files:
            return None

        prompt = (
            "Decide whether the question restricts the search to a particular "
            "document or section. Respond with JSON only.\n"
            f"Available documents: {json.dumps(self.known_files)}\n"
            f"Available sections: {json.dumps(self.known_sections[:40])}\n\n"
            'If it restricts to a document, respond {"source_file": "<exact name>"}.\n'
            'If it restricts to a section, respond {"section": "<exact name>"}.\n'
            'If it does not restrict the search, respond {}.\n\n'
            f"Question: {question}\n\n"
            "JSON:"
        )

        try:
            raw = self.llm(prompt) or ""
        except Exception:
            return None

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None

        if not isinstance(parsed, dict):
            return None

        where = {}
        source_file = parsed.get("source_file")
        section = parsed.get("section")

        if isinstance(source_file, str) and source_file in self.known_files:
            where["source_file"] = source_file
        if isinstance(section, str) and section in self.known_sections:
            where["section"] = section

        # Chroma needs an explicit $and for more than one condition.
        if len(where) > 1:
            return {"$and": [{k: v} for k, v in where.items()]}
        return where or None

    # -------------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------------
    def transform(self, question: str, chat_history=(), settings=None) -> TransformedQuery:
        """
        Apply the enabled transformations in order.

        Args:
            question: The user's question as typed
            chat_history: Previous (question, answer) pairs
            settings: RetrievalSettings carrying the on/off switches

        Returns:
            TransformedQuery; `search_query` is always usable, falling back to
            the original question whenever a transformation is unusable.
        """
        result = TransformedQuery(search_query=question)
        current = question

        condense_on = getattr(settings, "condense_history", True)
        rewrite_on = getattr(settings, "rewrite_query", False)
        multi_on = getattr(settings, "multi_query", False)
        hyde_on = getattr(settings, "hyde", False)
        filter_on = getattr(settings, "infer_filter", False)

        if condense_on:
            condensed = self.condense(current, chat_history)
            if condensed:
                result.condensed_query = condensed
                current = condensed
                result.notes.append("Resolved references against the conversation.")

        if rewrite_on:
            rewritten = self.rewrite(current)
            if rewritten:
                result.rewritten_query = rewritten
                current = rewritten
                result.notes.append("Rewrote the query for retrieval.")

        if multi_on:
            result.variants = self.multi_query(current)
            if result.variants:
                result.notes.append(
                    f"Searched {len(result.variants) + 1} phrasings of the question."
                )

        if hyde_on:
            hypothetical = self.hyde(current)
            if hypothetical:
                result.hyde_text = hypothetical
                result.notes.append("Searched using a hypothetical answer (HyDE).")

        if filter_on:
            where = self.infer_filter(question)
            if where:
                result.where_filter = where
                result.notes.append(f"Inferred metadata filter: {where}")

        result.search_query = current
        return result
