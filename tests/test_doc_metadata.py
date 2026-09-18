"""Tests for fingerprints, DOI detection and publication-detail extraction."""

import datetime
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library_fakes import details_llm, silent_llm
from rag_core import doc_metadata as dm


class TestFingerprints:
    def test_file_hash_is_stable_and_distinct(self):
        assert dm.file_sha256(b"abc") == dm.file_sha256(b"abc")
        assert dm.file_sha256(b"abc") != dm.file_sha256(b"abd")

    def test_text_hash_ignores_spacing_case_and_punctuation(self):
        a = "The Quick, brown fox.\n\nJumps!"
        b = "the quick brown   fox jumps"
        assert dm.text_sha256(a) == dm.text_sha256(b)
        assert dm.text_sha256(a) != dm.text_sha256("the quick brown cat jumps")

    def test_normalize_title(self):
        assert dm.normalize_title("  A Study: of FOXES! ") == "a study of foxes"


class TestDoi:
    @pytest.mark.parametrize("text, expected", [
        ("Published as doi:10.1038/s41586-020-2649-2 in Nature.", "10.1038/s41586-020-2649-2"),
        ("https://doi.org/10.1016/j.cell.2021.01.001.", "10.1016/j.cell.2021.01.001"),
        ("(DOI: 10.1371/journal.pone.0123456)", "10.1371/journal.pone.0123456"),
        ("No identifier in this text at all.", None),
    ])
    def test_find_doi(self, text, expected):
        assert dm.find_doi(text) == expected

    def test_find_doi_only_searches_the_front(self):
        text = "x" * 30000 + " 10.1000/late.doi"
        assert dm.find_doi(text) is None


class TestFileProperties:
    def _docx(self, title, author, created):
        import docx
        document = docx.Document()
        document.add_paragraph("Body text.")
        document.core_properties.title = title
        document.core_properties.author = author
        document.core_properties.created = created
        buf = io.BytesIO()
        document.save(buf)
        return buf.getvalue()

    def test_docx_core_properties(self):
        data = self._docx("Fox Behaviour", "Smith, J.", datetime.datetime(2019, 5, 1))
        props = dm.read_file_properties(data, ".docx")
        assert props["title"] == "Fox Behaviour"
        assert props["authors"] == "Smith, J."
        assert props["year"] == "2019"

    @pytest.mark.parametrize("junk", [
        "Microsoft Word - draft3.docx", "untitled", "paper_final.pdf", "   ",
    ])
    def test_junk_titles_dropped(self, junk):
        data = self._docx(junk, "", datetime.datetime(2019, 5, 1))
        assert "title" not in dm.read_file_properties(data, ".docx")

    def test_unreadable_bytes_give_empty_dict(self):
        assert dm.read_file_properties(b"not a real file", ".pdf") == {}


class TestLlmExtraction:
    def test_parses_fenced_json(self):
        got = dm.extract_details_llm("First page text", details_llm(doi="10.1/x"))
        assert got == {"title": "A Study of Foxes", "authors": "Smith, J.; Lee, K.",
                       "year": "2021", "journal": "J. Canid Res.", "doi": "10.1/x"}

    def test_garbage_or_failure_gives_empty(self):
        assert dm.extract_details_llm("text", silent_llm) == {}

        def boom(prompt):
            raise RuntimeError("model down")
        assert dm.extract_details_llm("text", boom) == {}

    def test_no_llm_or_no_text(self):
        assert dm.extract_details_llm("text", None) == {}
        assert dm.extract_details_llm("   ", details_llm()) == {}


class TestMerge:
    def test_precedence(self):
        merged = dm.merge_details(
            doi_text="10.1/from-text",
            file_props={"title": "File Title", "authors": "File Author", "year": "2018",
                        "doi": "10.1/from-file"},
            llm={"title": "LLM Title", "year": "2021", "journal": "J", "doi": "10.1/from-llm"},
        )
        assert merged["doi"] == dm.Detail("10.1/from-text", "doi_text")
        assert merged["title"] == dm.Detail("LLM Title", "llm")
        assert merged["year"] == dm.Detail("2021", "llm")
        assert merged["authors"] == dm.Detail("File Author", "file_properties")
        assert merged["journal"] == dm.Detail("J", "llm")

    def test_fallbacks_and_invalid_year(self):
        merged = dm.merge_details(None, {"doi": "10.1/f"}, {"year": "circa 2020s"})
        assert merged["doi"] == dm.Detail("10.1/f", "file_properties")
        assert merged["year"] == dm.Detail("", "none")
        assert merged["title"] == dm.Detail("", "none")

    def test_llm_doi_beats_file_doi(self):
        merged = dm.merge_details(None, {"doi": "10.1/f"}, {"doi": "10.1/l"})
        assert merged["doi"] == dm.Detail("10.1/l", "llm")


def test_first_page_text_uses_first_blocks():
    blocks = [{"text": "Title page", "page": 1}, {"text": "More", "page": 1},
              {"text": "Page two", "page": 2}]
    assert dm.first_page_text(blocks) == "Title page\n\nMore"


def test_page_or_slide_count_from_blocks():
    assert dm.page_or_slide_count([{"page": 1}, {"page": 7}, {"page": 3}]) == 7
    assert dm.page_or_slide_count([{"slide": 4}]) == 4
    assert dm.page_or_slide_count([{"text": "docx"}]) == 0
