"""
Unit tests for the retrieval engine.

`rag_core` deliberately has no Streamlit dependency, so everything here runs
without a browser or a running app. Nothing in this file needs Ollama either:
embeddings are stubbed, which keeps the suite fast and deterministic. The
end-to-end behaviour that *does* need a live model is exercised manually --
see the Verification section of ADVANCED_RAG_PLAN.md.

Run with:  pytest tests/ -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.chunker import Chunk, chunk_document, chunk_documents, _split_text
from rag_core.citations import verify_citations
from rag_core.context_assembler import assemble_context, context_budget_chars
from rag_core.query_transformer import QueryTransformer, _clean_single_line, looks_dependent
from rag_core.reranker import LLMReranker
from rag_core.store import EphemeralRagStore, RetrievedChunk, tokenize
from rag_core.context_assembler import Source


# -----------------------------------------------------------------------------
# Test doubles
# -----------------------------------------------------------------------------
class StubEmbedder:
    """
    Deterministic embeddings with no network.

    Vectors are bag-of-characters counts, which is meaningless semantically but
    gives stable, comparable numbers -- enough to exercise ranking and fusion.
    """

    model = "stub-embed"

    def __init__(self):
        self.dimension = 26

    def _vector(self, text):
        vec = [0.0] * 26
        for char in text.lower():
            if "a" <= char <= "z":
                vec[ord(char) - 97] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts, progress_callback=None):
        vectors = [self._vector(t) for t in texts]
        if progress_callback:
            progress_callback(len(texts), len(texts))
        return vectors

    def embed_query(self, text):
        return self._vector(text)


def make_blocks(n=3, text="The quick brown fox jumps over the lazy dog repeatedly."):
    """
    Blocks that share common wording but each carry a unique rare token.

    The distinct tokens matter: BM25 scores a term appearing in *every*
    document at or below zero, so a corpus of identical text gives the sparse
    index nothing to discriminate on.
    """
    return [
        {"text": f"{text} Marker ZQ{i}X here.", "page": i + 1, "slide": None,
         "section": f"Section {i}", "block_index": i}
        for i in range(n)
    ]


def make_hit(index, parent, text="body text here", size=400):
    chunk = Chunk(
        chunk_id=f"c{index}", text=text, parent_id=parent,
        parent_text="P" * size, source_file="doc.pdf", doc_type=".pdf",
        chunk_index=index, char_start=0, char_end=len(text), page=index + 1,
    )
    return RetrievedChunk(chunk=chunk, fused_score=1.0 / (index + 1), dense_rank=index)


# -----------------------------------------------------------------------------
# Chunker
# -----------------------------------------------------------------------------
class TestChunker:
    def test_split_covers_all_text_without_gaps(self):
        text = "\n\n".join(f"Paragraph {i} with several words in it." for i in range(30))
        spans = _split_text(text, 200, 40)

        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)
        assert all(spans[i][0] < spans[i + 1][0] for i in range(len(spans) - 1))

    def test_split_overlaps_by_requested_amount(self):
        text = "word " * 400
        spans = _split_text(text, 300, 50)
        overlaps = [spans[i][1] - spans[i + 1][0] for i in range(len(spans) - 1)]

        assert all(o == 50 for o in overlaps)

    def test_short_text_is_a_single_span(self):
        assert _split_text("short", 100, 10) == [(0, 5)]

    def test_chunks_never_straddle_a_page(self):
        chunks = chunk_document(make_blocks(4), "doc.pdf", ".pdf")

        assert len(chunks) == 4
        assert [c.page for c in chunks] == [1, 2, 3, 4]

    def test_chunk_ids_are_unique(self):
        chunks = chunk_document(make_blocks(20), "doc.pdf", ".pdf")

        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_children_are_contained_in_their_parent(self):
        long_block = [{"text": "sentence about measurements. " * 400, "page": 1,
                       "slide": None, "section": "Results", "block_index": 0}]
        chunks = chunk_document(long_block, "doc.pdf", ".pdf")

        assert len(chunks) > 1
        assert all(c.text.strip() in c.parent_text for c in chunks)

    def test_metadata_is_chroma_safe(self):
        """Chroma stores scalars only and rejects None."""
        metadata = chunk_document(make_blocks(1), "doc.pdf", ".pdf")[0].to_metadata()

        assert all(isinstance(v, (str, int, float, bool)) for v in metadata.values())
        assert None not in metadata.values()

    def test_absent_page_and_slide_become_zero_not_none(self):
        blocks = [{"text": "some body text for a word document", "page": None,
                   "slide": None, "section": None, "block_index": 0}]
        metadata = chunk_document(blocks, "doc.docx", ".docx")[0].to_metadata()

        assert metadata["page"] == 0
        assert metadata["slide"] == 0
        assert metadata["section"] == ""

    def test_same_filename_twice_does_not_collide(self):
        """Two uploads can share a name; their chunk ids must still differ."""
        results = [
            {"filename": "same.pdf", "file_type": ".pdf",
             "blocks": make_blocks(2), "error": None}
            for _ in range(2)
        ]
        chunks = chunk_documents(results)

        assert len(chunks) == 4
        assert len({c.chunk_id for c in chunks}) == 4

    def test_empty_blocks_produce_no_chunks(self):
        assert chunk_document([], "doc.pdf", ".pdf") == []
        assert chunk_document([{"text": "   ", "block_index": 0}], "d.pdf", ".pdf") == []


# -----------------------------------------------------------------------------
# Tokenizer / BM25 input
# -----------------------------------------------------------------------------
class TestTokenize:
    def test_keeps_identifiers_intact(self):
        """Rare literal tokens are exactly what the sparse index exists for."""
        tokens = tokenize("The CFTR-KO line showed a 3.2-fold rise, lot ZZQX7-4419.")

        assert "cftr-ko" in tokens
        assert "3.2-fold" in tokens
        assert "zzqx7-4419" in tokens

    def test_drops_stopwords(self):
        assert tokenize("what happens when a cell decides to destroy itself") == [
            "happens", "cell", "decides", "destroy"
        ]

    def test_all_stopword_query_yields_nothing(self):
        """The sparse index should abstain rather than rank on grammar."""
        assert tokenize("what is it about") == []


# -----------------------------------------------------------------------------
# Store: fusion, filtering, lifecycle
# -----------------------------------------------------------------------------
class TestStore:
    @pytest.fixture
    def store(self):
        s = EphemeralRagStore(StubEmbedder())
        s.add_chunks(chunk_document(make_blocks(5), "doc.pdf", ".pdf"))
        yield s
        s.close()

    def test_ingest_populates_both_indexes(self, store):
        assert store.count == 5
        assert not store.is_empty
        assert store.sparse_search("zq3x", k=5)

    def test_sparse_finds_the_chunk_holding_a_rare_token(self, store):
        """The whole reason for keeping a keyword index alongside the vectors."""
        hits = store.sparse_search("zq3x", k=5)

        assert hits
        assert "ZQ3X" in store.get_chunk(hits[0][0]).text

    def test_sparse_abstains_on_a_term_present_everywhere(self, store):
        """
        BM25 scores a term appearing in every document at or below zero: it
        carries no discriminating power. Abstaining leaves the ranking to the
        dense index rather than contributing arbitrary noise.
        """
        assert store.sparse_search("fox", k=5) == []

    def test_rrf_scores_match_hand_computation(self, store):
        """
        RRF weight for rank r is  weight / (k + r + 1).

        A chunk at rank 0 in both indexes with dense_weight 0.5 and k=60 scores
        0.5/61 + 0.5/61.
        """
        dense = [("a", 0.9), ("b", 0.8)]
        sparse = [("a", 5.0), ("c", 1.0)]
        store._chunks = {
            cid: Chunk(chunk_id=cid, text="t", parent_id="p", parent_text="t",
                       source_file="d.pdf", doc_type=".pdf", chunk_index=0,
                       char_start=0, char_end=1)
            for cid in ("a", "b", "c")
        }

        fused = store._fuse(dense, sparse, fused_k=3, dense_weight=0.5, rrf_k=60)
        by_id = {r.chunk.chunk_id: r.fused_score for r in fused}

        assert by_id["a"] == pytest.approx(0.5 / 61 + 0.5 / 61)
        assert by_id["b"] == pytest.approx(0.5 / 62)
        assert by_id["c"] == pytest.approx(0.5 / 62)
        assert fused[0].chunk.chunk_id == "a"

    def test_dense_weight_extremes_disable_one_index(self, store):
        dense = [("a", 0.9)]
        sparse = [("b", 5.0)]
        store._chunks = {
            cid: Chunk(chunk_id=cid, text="t", parent_id="p", parent_text="t",
                       source_file="d.pdf", doc_type=".pdf", chunk_index=0,
                       char_start=0, char_end=1)
            for cid in ("a", "b")
        }

        keyword_only = {r.chunk.chunk_id: r.fused_score
                        for r in store._fuse(dense, sparse, 2, dense_weight=0.0, rrf_k=60)}
        semantic_only = {r.chunk.chunk_id: r.fused_score
                         for r in store._fuse(dense, sparse, 2, dense_weight=1.0, rrf_k=60)}

        assert keyword_only["a"] == 0.0
        assert semantic_only["b"] == 0.0

    def test_retrieval_source_is_reported(self, store):
        hits = store.search("fox", dense_k=5, sparse_k=5, fused_k=5)

        assert hits
        assert all(h.retrieval_source in {"both", "semantic", "keyword"} for h in hits)

    def test_metadata_filter_restricts_results(self):
        s = EphemeralRagStore(StubEmbedder())
        s.add_chunks(chunk_document(make_blocks(2), "a.pdf", ".pdf"))
        s.add_chunks(chunk_document(make_blocks(2), "b.pdf", ".pdf"))

        hits = s.search("fox", where={"source_file": "b.pdf"}, fused_k=10)

        assert hits
        assert {h.chunk.source_file for h in hits} == {"b.pdf"}
        s.close()

    def test_clear_empties_every_index(self, store):
        store.clear()

        assert store.count == 0
        assert store.is_empty
        assert store.search("fox") == []
        assert store.sparse_search("fox") == []

    def test_store_is_reusable_after_clear(self, store):
        store.clear()
        added = store.add_chunks(chunk_document(make_blocks(3), "new.pdf", ".pdf"))

        assert added == 3
        assert store.count == 3

    def test_two_stores_coexist(self):
        """Chroma's ephemeral client is a process-wide singleton."""
        a = EphemeralRagStore(StubEmbedder())
        b = EphemeralRagStore(StubEmbedder())

        assert a.collection_name != b.collection_name
        a.add_chunks(chunk_document(make_blocks(2), "a.pdf", ".pdf"))
        assert b.count == 0

        a.close()
        b.close()

    def test_duplicate_chunks_are_not_reindexed(self, store):
        chunks = chunk_document(make_blocks(5), "doc.pdf", ".pdf")

        assert store.add_chunks(chunks) == 0
        assert store.count == 5

    def test_empty_store_searches_return_empty(self):
        s = EphemeralRagStore(StubEmbedder())
        assert s.search("anything") == []
        s.close()


# -----------------------------------------------------------------------------
# Context assembly -- the truncation fix
# -----------------------------------------------------------------------------
class TestContextAssembler:
    def test_budget_scales_with_context_window(self):
        assert context_budget_chars(2048) == int(2048 * 0.6 * 4)
        assert context_budget_chars(8192) > context_budget_chars(2048)

    def test_never_exceeds_budget(self):
        """The whole point: overflow must be impossible, not merely unlikely."""
        hits = [make_hit(i, f"p{i}", size=2000) for i in range(50)]
        ctx = assemble_context(hits, num_ctx=1024)

        assert ctx.used_chars <= ctx.budget_chars

    def test_reports_what_it_dropped(self):
        hits = [make_hit(i, f"p{i}", size=2000) for i in range(50)]
        ctx = assemble_context(hits, num_ctx=1024)

        assert ctx.dropped_count > 0
        assert len(ctx.sources) + ctx.dropped_count == len(hits)

    def test_falls_back_to_child_text_when_parent_will_not_fit(self):
        hits = [make_hit(0, "p0", text="short child", size=9000)]
        ctx = assemble_context(hits, num_ctx=2048)

        assert len(ctx.sources) == 1
        assert ctx.sources[0].text == "short child"

    def test_deduplicates_shared_parents(self):
        hits = [make_hit(0, "shared"), make_hit(1, "shared"), make_hit(2, "other")]
        ctx = assemble_context(hits, num_ctx=8192)

        assert len(ctx.sources) == 2

    def test_source_numbering_is_contiguous(self):
        hits = [make_hit(i, f"p{i}", size=2000) for i in range(50)]
        ctx = assemble_context(hits, num_ctx=1024)

        assert [s.index for s in ctx.sources] == list(range(1, len(ctx.sources) + 1))

    def test_markers_appear_in_the_context_text(self):
        ctx = assemble_context([make_hit(i, f"p{i}") for i in range(3)], num_ctx=8192)

        for source in ctx.sources:
            assert source.marker in ctx.text

    def test_empty_input(self):
        ctx = assemble_context([], num_ctx=8192)

        assert ctx.is_empty
        assert ctx.text == ""


# -----------------------------------------------------------------------------
# Citations
# -----------------------------------------------------------------------------
class TestCitations:
    @pytest.fixture
    def sources(self):
        return [Source(index=i, chunk_id=f"c{i}", text="", location_label=f"d.pdf p.{i}",
                       source_file="d.pdf") for i in (1, 2, 3)]

    def test_detects_fabricated_marker(self, sources):
        report = verify_citations("Cooled to 4C [S1]. Pressure held at 3 bar [S9].", sources)

        assert report.hallucinated == {9}
        assert not report.is_clean

    def test_clean_answer_passes(self, sources):
        report = verify_citations("Cooled to 4C [S1]. Yield rose 12 percent [S2].", sources)

        assert report.is_clean
        assert report.cited_indexes == {1, 2}
        assert report.coverage == 1.0

    def test_flags_uncited_claims(self, sources):
        report = verify_citations(
            "Cooled to 4C [S1]. The pressure was held at three bar for the whole run.",
            sources,
        )

        assert report.is_clean
        assert len(report.uncited_sentences) == 1

    def test_decimals_do_not_split_sentences(self, sources):
        report = verify_citations(
            "The constant was 0.0472 volts per liter per Smith et al. during setup [S3].",
            sources,
        )

        assert report.total_claim_sentences == 1
        assert not report.uncited_sentences

    def test_refusal_is_not_flagged(self, sources):
        report = verify_citations("This information is not found in the provided documents.", sources)

        assert report.is_clean
        assert "No factual claims" in report.summary()

    def test_bulleted_answers_are_checked_per_line(self, sources):
        report = verify_citations(
            "Findings:\n- CFTR-KO line measured 38 nA per square cm [S1]\n"
            "- Wildtype control measured 100 nA per square cm [S2]",
            sources,
        )

        assert report.cited_indexes == {1, 2}
        assert not report.uncited_sentences


# -----------------------------------------------------------------------------
# Query transformation
# -----------------------------------------------------------------------------
class TestQueryTransformer:
    @pytest.mark.parametrize("raw,expected", [
        ("Sure, here is the rewritten query: reactor temperature", "reactor temperature"),
        ('"reactor temperature"', "reactor temperature"),
        ("1. reactor temperature", "reactor temperature"),
        ("Here's the standalone question:\nWhat was the temperature?", "What was the temperature?"),
        ("<think>reasoning</think>\nreactor temperature", "reactor temperature"),
        ("I cannot help with that", None),
        ("", None),
        ("x" * 500, None),
    ])
    def test_sanitizer(self, raw, expected):
        assert _clean_single_line(raw) == expected

    @pytest.mark.parametrize("query", [
        "What does the study say about the answer: yield?",
        "Table 3: fold change values",
    ])
    def test_sanitizer_preserves_legitimate_colons(self, query):
        assert _clean_single_line(query) == query

    def test_dependency_heuristic(self):
        assert looks_dependent("How does that compare?")
        assert looks_dependent("why?")
        assert not looks_dependent("Explain the CFTR mutation mechanism in detail")

    def test_condense_skipped_without_history(self):
        """No history means nothing to resolve against -- and no call to spend."""
        calls = []
        t = QueryTransformer(lambda p: calls.append(p) or "x")

        assert t.condense("How does that compare?", []) is None
        assert calls == []

    def test_filter_inference_rejects_unknown_file(self):
        t = QueryTransformer(lambda p: '{"source_file": "ghost.pdf"}',
                             known_files=["real.pdf"])

        assert t.infer_filter("what does ghost.pdf say?") is None

    def test_filter_inference_accepts_known_file(self):
        t = QueryTransformer(lambda p: '{"source_file": "real.pdf"}',
                             known_files=["real.pdf"])

        assert t.infer_filter("what does real.pdf say?") == {"source_file": "real.pdf"}

    def test_transform_falls_back_when_llm_fails(self):
        def broken(prompt):
            raise RuntimeError("model unavailable")

        class Settings:
            condense_history = True
            rewrite_query = True
            multi_query = True
            hyde = True
            infer_filter = True

        result = QueryTransformer(broken, known_files=["a.pdf"]).transform(
            "What is the yield?", [("prior", "answer")], Settings()
        )

        assert result.search_query == "What is the yield?"
        assert result.variants == []
        assert result.where_filter is None


# -----------------------------------------------------------------------------
# Re-ranking
# -----------------------------------------------------------------------------
class TestReranker:
    def test_reorders_by_score(self):
        hits = [make_hit(i, f"p{i}") for i in range(4)]
        out = LLMReranker(lambda p: '{"0": 1, "1": 9, "2": 0, "3": 7}')("q", hits, top_k=3)

        assert [h.chunk.chunk_id for h in out] == ["c1", "c3", "c0"]

    def test_clamps_out_of_range_scores(self):
        hits = [make_hit(i, f"p{i}") for i in range(2)]
        out = LLMReranker(lambda p: '{"0": 500, "1": -20}')("q", hits, top_k=2)

        assert out[0].rerank_score == 10.0
        assert out[1].rerank_score == 0.0

    @pytest.mark.parametrize("llm", [
        lambda p: "I don't know",
        lambda p: "",
        lambda p: (_ for _ in ()).throw(RuntimeError("model down")),
    ])
    def test_falls_back_to_retrieval_order(self, llm):
        """A broken reranker must never be worse than no reranker."""
        hits = [make_hit(i, f"p{i}") for i in range(4)]
        out = LLMReranker(llm)("q", hits, top_k=4)

        assert [h.chunk.chunk_id for h in out] == ["c0", "c1", "c2", "c3"]

    def test_empty_candidates(self):
        assert LLMReranker(lambda p: "{}")("q", [], top_k=5) == []
