"""
Citation verification.

Asking a model to cite its sources is easy; checking that it did is what makes
the citations worth anything. A fabricated `[S7]` next to a claim looks exactly
as authoritative as a real one, and the previous version of this app had no way
to tell them apart.

The checks here are deterministic and free -- no model call. They cannot judge
whether a cited passage genuinely supports a claim, but they catch the failures
that matter most in practice: markers pointing at sources that were never
supplied, and substantive claims carrying no citation at all.
"""

import re
from dataclasses import dataclass, field
from typing import List, Sequence, Set

from .context_assembler import Source

_MARKER_RE = re.compile(r"\[S(\d+)\]")

# Sentence splitter that leaves decimals, abbreviations and initials intact --
# "0.0472 volts" and "et al." must not be read as sentence boundaries.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“])")

# Sentences that make no factual claim need no citation.
_NON_CLAIM_PREFIXES = (
    "here", "in summary", "in short", "to summarize", "overall",
    "this information is not found", "no documents", "however", "note that",
)


@dataclass
class CitationReport:
    """The result of checking one answer's citations."""

    cited_indexes: Set[int] = field(default_factory=set)
    hallucinated: Set[int] = field(default_factory=set)
    unused_sources: Set[int] = field(default_factory=set)
    uncited_sentences: List[str] = field(default_factory=list)
    total_claim_sentences: int = 0

    @property
    def is_clean(self) -> bool:
        """True when no citation points at a source that was never supplied."""
        return not self.hallucinated

    @property
    def coverage(self) -> float:
        """Share of claim sentences carrying at least one citation."""
        if not self.total_claim_sentences:
            return 1.0
        cited = self.total_claim_sentences - len(self.uncited_sentences)
        return cited / self.total_claim_sentences

    def summary(self) -> str:
        """One-line verdict for the UI."""
        if self.hallucinated:
            markers = ", ".join(f"[S{i}]" for i in sorted(self.hallucinated))
            return f"⚠️ Cites {markers}, which was never provided."
        if not self.cited_indexes:
            # An answer that asserts nothing -- a refusal, say -- has nothing
            # to cite, and flagging it would train the user to ignore warnings.
            if not self.total_claim_sentences:
                return "✅ No factual claims to cite."
            return "⚠️ No citations in this answer."
        return f"✅ {len(self.cited_indexes)} source(s) cited · {self.coverage:.0%} of claims cited"


def extract_markers(text: str) -> Set[int]:
    """Source numbers cited anywhere in the text."""
    return {int(m) for m in _MARKER_RE.findall(text or "")}


def _is_claim(sentence: str) -> bool:
    """Whether a sentence asserts something that ought to be cited."""
    stripped = sentence.strip()

    # Too short to carry a claim, or structural (headings, list scaffolding).
    if len(stripped) < 25:
        return False
    if stripped.endswith(":"):
        return False
    if stripped.lower().startswith(_NON_CLAIM_PREFIXES):
        return False
    return True


def verify_citations(answer: str, sources: Sequence[Source]) -> CitationReport:
    """
    Check an answer's citations against the sources actually supplied.

    Args:
        answer: The model's answer text
        sources: The sources given to the model for this answer

    Returns:
        CitationReport naming any fabricated markers and uncited claims
    """
    report = CitationReport()

    if not answer:
        return report

    valid = {source.index for source in sources}
    report.cited_indexes = extract_markers(answer)
    report.hallucinated = report.cited_indexes - valid
    report.unused_sources = valid - report.cited_indexes

    # Bulleted answers put each claim on its own line, which the sentence
    # splitter would otherwise merge into one run-on sentence.
    for line in answer.split("\n"):
        for sentence in _SENTENCE_RE.split(line):
            if not _is_claim(sentence):
                continue
            report.total_claim_sentences += 1
            if not _MARKER_RE.search(sentence):
                report.uncited_sentences.append(sentence.strip())

    return report


def strip_markers(text: str) -> str:
    """Remove citation markers, e.g. for a clean copy of the prose."""
    return re.sub(r"\s*\[S\d+\]", "", text or "").strip()
