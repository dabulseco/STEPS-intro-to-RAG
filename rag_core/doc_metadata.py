"""
Document fingerprints and publication details.

Used by persistent libraries to (1) recognise a document that is already in the
library and (2) show what the library contains without keeping the files.

Details come from three offline sources, in rising order of trust for most
fields: properties stored inside the file, the local LLM reading the first
page, and -- for the DOI only -- a DOI printed in the text itself. A user's
manual edit overrides all of them (handled by the caller). No network lookup
is made.
"""

import datetime
import hashlib
import io
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

from .eval import parse_metric_payload
from .library_db import DETAIL_FIELDS, normalize_title

__all__ = [
    "Detail", "file_sha256", "text_sha256", "normalize_text_for_hash",
    "normalize_title", "find_doi", "read_file_properties", "first_page_text",
    "page_or_slide_count", "extract_details_llm", "merge_details",
]

# 10.<registrant>/<suffix>. The suffix is permissive by design; trailing
# punctuation picked up from the surrounding sentence is trimmed afterwards.
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")
_DOI_TRAILING = ".,;:)]}'\""

# DOIs are printed on the first page or two; searching further mostly finds
# DOIs of *cited* papers in the reference list.
_DOI_SEARCH_CHARS = 20000

_JUNK_TITLE = re.compile(
    r"^(microsoft (word|powerpoint) - .*|untitled.*|.*\.(pdf|docx?|pptx?|tex|txt))$",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"^(1[5-9]\d\d|20\d\d)$")

_FIRST_PAGE_CHARS = 3000


@dataclass(frozen=True)
class Detail:
    """One publication detail and where it came from."""

    value: str
    source: str   # 'doi_text' | 'file_properties' | 'llm' | 'user' | 'none'


# -----------------------------------------------------------------------------
# Fingerprints
# -----------------------------------------------------------------------------
def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text_for_hash(text: str) -> str:
    """Lowercase letters and digits only: survives re-saving and re-flowing."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text_for_hash(text).encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# DOI and file properties
# -----------------------------------------------------------------------------
def find_doi(text: str, max_chars: int = _DOI_SEARCH_CHARS) -> Optional[str]:
    """The first DOI in the front of the text, or None."""
    match = _DOI_RE.search((text or "")[:max_chars])
    if not match:
        return None
    doi = match.group(1).rstrip(_DOI_TRAILING)
    # A closing parenthesis is legitimate only if the DOI opened one.
    while doi.endswith(")") and doi.count("(") < doi.count(")"):
        doi = doi[:-1]
    return doi or None


def _clean_title(title) -> Optional[str]:
    title = " ".join(str(title or "").split())
    if not title or _JUNK_TITLE.match(title):
        return None
    return title


def _year_of(value) -> Optional[str]:
    if isinstance(value, (datetime.date, datetime.datetime)):
        return str(value.year)
    match = re.search(r"(1[5-9]\d\d|20\d\d)", str(value or ""))
    return match.group(1) if match else None


def _pdf_properties(data: bytes) -> Dict[str, str]:
    import PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(data))
    info = reader.metadata or {}
    props: Dict[str, str] = {}
    title = _clean_title(info.get("/Title"))
    if title:
        props["title"] = title
    author = " ".join(str(info.get("/Author") or "").split())
    if author:
        props["authors"] = author
    # PDF dates look like "D:20210315120000+00'00'".
    year = _year_of(info.get("/CreationDate"))
    if year:
        props["year"] = year
    for key in ("/doi", "/DOI", "/Subject"):
        doi = find_doi(str(info.get(key) or ""))
        if doi:
            props["doi"] = doi
            break
    return props


def _office_properties(core) -> Dict[str, str]:
    props: Dict[str, str] = {}
    title = _clean_title(core.title)
    if title:
        props["title"] = title
    author = " ".join(str(core.author or "").split())
    if author:
        props["authors"] = author
    year = _year_of(core.created)
    if year:
        props["year"] = year
    return props


def read_file_properties(data: bytes, ext: str) -> Dict[str, str]:
    """
    Title, authors, year and DOI stored inside the file, where present.

    Never raises: an unreadable file simply has no properties. Junk titles
    ("Microsoft Word - draft3.docx", "untitled", bare filenames) are dropped.
    """
    ext = (ext or "").lower()
    try:
        if ext == ".pdf":
            return _pdf_properties(data)
        if ext in (".docx", ".doc"):
            import docx
            return _office_properties(docx.Document(io.BytesIO(data)).core_properties)
        if ext == ".pptx":
            import pptx
            return _office_properties(pptx.Presentation(io.BytesIO(data)).core_properties)
    except Exception:
        return {}
    return {}


# -----------------------------------------------------------------------------
# Block helpers
# -----------------------------------------------------------------------------
def first_page_text(blocks: Sequence[Dict], max_chars: int = _FIRST_PAGE_CHARS) -> str:
    """Text of the first page/slide (or the opening of a DOCX), capped."""
    if not blocks:
        return ""
    first = blocks[0]
    key = "page" if first.get("page") else ("slide" if first.get("slide") else None)
    parts = []
    for block in blocks:
        if key and block.get(key) != first.get(key):
            break
        parts.append(str(block.get("text", "")).strip())
        if not key and sum(len(p) for p in parts) >= max_chars:
            break
    return "\n\n".join(p for p in parts if p)[:max_chars]


def page_or_slide_count(blocks: Sequence[Dict]) -> int:
    """Highest page or slide number seen; 0 for formats without pages."""
    numbers = [b.get("page") or b.get("slide") or 0 for b in blocks]
    return max(numbers, default=0)


# -----------------------------------------------------------------------------
# LLM extraction
# -----------------------------------------------------------------------------
_LLM_PROMPT = """Extract the bibliographic details of this document from its first page.

Return ONLY a JSON object with these keys:
{{"title": "...", "authors": "...", "year": "...", "journal": "...", "doi": "..."}}

Rules:
- authors: names separated by "; " as printed.
- year: four digits.
- journal: journal, conference or publisher name.
- Use "" for anything not shown on the page. Do not guess.

FIRST PAGE:
{text}
"""


def extract_details_llm(
    first_page: str, ask_llm: Optional[Callable[[str], str]]
) -> Dict[str, str]:
    """Ask the local LLM for publication details; {} on any failure."""
    if ask_llm is None or not (first_page or "").strip():
        return {}
    try:
        raw = ask_llm(_LLM_PROMPT.format(text=first_page[:_FIRST_PAGE_CHARS]))
        payload = parse_metric_payload(raw)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, str] = {}
    for key in DETAIL_FIELDS:
        value = " ".join(str(payload.get(key) or "").split())
        if value:
            out[key] = value
    return out


# -----------------------------------------------------------------------------
# Precedence
# -----------------------------------------------------------------------------
def merge_details(
    doi_text: Optional[str],
    file_props: Dict[str, str],
    llm: Dict[str, str],
) -> Dict[str, Detail]:
    """
    Combine the sources into one value per field.

    DOI: text > LLM > file properties. Other fields: LLM > file properties,
    because stored titles are so often junk. Years that are not four digits
    are discarded.
    """
    merged: Dict[str, Detail] = {}
    for key in DETAIL_FIELDS:
        candidates = []
        if key == "doi" and doi_text:
            candidates.append(Detail(doi_text, "doi_text"))
        if llm.get(key):
            candidates.append(Detail(llm[key], "llm"))
        if file_props.get(key):
            candidates.append(Detail(file_props[key], "file_properties"))
        if key == "year":
            candidates = [c for c in candidates if _YEAR_RE.match(c.value)]
        merged[key] = candidates[0] if candidates else Detail("", "none")
    return merged
