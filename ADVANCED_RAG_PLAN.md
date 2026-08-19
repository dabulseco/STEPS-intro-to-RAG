# Advanced RAG Upgrade — Local LLM Scientific Assistant

## Context

**Why this change.** The app is marketed as having a "strict RAG mode," but there is no retrieval
layer anywhere in it. `grep -rniE 'embed|vector|chroma|faiss|chunk|bm25|rerank|retriev|splitter'`
returns zero hits in code. What exists is **long-context prompt stuffing**: every uploaded document
is flattened to plain text, concatenated into one string (`st.session_state.document_content`), and
pasted into *every* query along with an instruction to stay grounded.

That design has a hard failure mode. `num_ctx` maxes at **8192** and defaults to **2048**
(`app.py:263`), while uploads are allowed up to **200 MB per file**
(`utils/document_processor.py:19`). A single 20-page paper already exceeds the ceiling, so Ollama
silently truncates and the model never sees most of the document — then confidently answers from a
fragment, or reports "not found in the provided document." Users read this as "the model is bad."

**Intended outcome.** Insert a real retrieval pipeline between extraction and prompting —
`parse → chunk (+metadata) → embed → index → transform query → hybrid retrieve → optional re-rank →
token-budgeted assembly with citations` — keeping the existing Perspective/Audience system, the
STRICT / HYBRID / LLM toggle, templates, and HTML export intact.

**Ephemerality is a requirement, not a limitation.** The vector store lives in memory for the
lifetime of the process, is clearable on demand, and dies at shutdown. A separate persistent-chatbot
app exists and will absorb these features later, which imposes a design constraint: **the retrieval
modules must be storage-agnostic**, so swapping `EphemeralClient()` → `PersistentClient(path=...)`
is a one-line change with no other edits.

---

## How ephemerality reshapes the design

| Concern | Effect |
|---|---|
| Persistence, doc versioning, upsert/dedup across sessions | **Dropped entirely.** Large simplification — no migration, no stale-index reconciliation. |
| BM25 ↔ vector index sync | **Non-issue.** Both are rebuilt together from the same chunk list at ingest. In a persistent design this is the hardest part; here it's free. |
| Ingest cost | **Paid every session.** Makes embedding throughput a first-class UX concern → fast embedding model, batching, visible progress, and expensive ingest-time LLM enrichment kept **opt-in**. |
| Clear semantics | One `RagStore.clear()` drops the Chroma collection, the BM25 index, and the chunk list together — no partial-state bugs. |
| Portability to the persistent app | Store access goes through one thin class. Everything above it (chunking, fusion, transformation, re-ranking, assembly) is storage-independent and lifts over unchanged. |

---

## Architecture

### Build vs. rewrite decision

**Modify this app, but build the retrieval engine as standalone greenfield code.**

The broken surface is small and well-isolated. Of ~2,000 lines, only ~12% changes: the answer path
in `app.py` (~150 of 890 lines — `generate_ai_response` plus the two duplicated mode blocks),
`document_processor`'s output shape (~60 lines), and a small `prompt_builder` fix. Everything
expensive to recreate — the template system, the perspective/audience matrix, DOCX table
extraction, PPTX text-frame walking, the Bootstrap HTML exporter — is retrieval-agnostic and stays.

A full rewrite was considered and rejected: it would require re-deriving or copying all of that
content anyway, and it would leave **no working baseline to A/B against**. That matters
disproportionately here, because retrieval quality changes ("did re-ranking actually help?") cannot
be verified by inspection — only by comparison. Incremental phases keep a runnable app at every
step. Revisit this decision only if the app moves off Streamlit or gains multi-user requirements.

### Module layout

New code lives in a **standalone `rag_core/` package**, not in `utils/` (this app's junk drawer).
`rag_core` imports no Streamlit, touches no `session_state`, and knows nothing about the store's
lifetime beyond the instance injected into it. This app becomes its *first consumer* rather than
its owner — so porting to the persistent chatbot app is a directory copy plus a one-line
`EphemeralClient` → `PersistentClient` swap.

```
rag_core/                 [NEW — greenfield, no Streamlit, unit-tested]
  chunker.py              structure-aware splitting + metadata attachment
  embeddings.py           Ollama embedding client + model discovery/validation
  store.py                EphemeralRagStore: Chroma(in-mem) + BM25 + RRF + clear()
  query_transformer.py    condense / rewrite / multi-query / HyDE / filter inference
  reranker.py             LLM batch re-ranker (+ optional cross-encoder)
  context_assembler.py    token-budgeted packing, dedup, citation markers
  pipeline.py             single orchestrator — the ONLY entry point app.py calls

utils/                    [MODIFY — existing app-specific helpers]
  document_processor.py   emit structured blocks, not one flat string
  prompt_builder.py       actually use has_documents; add grounding-mode text
  template_manager.py     untouched
  html_export.py          untouched
```

`app.py` keeps all Streamlit code and calls `rag_core.pipeline.answer(...)` once. The two duplicated
mode-formatting blocks (`app.py:495-530` and `app.py:585-615`) collapse into that single call.

---

## Data-store-side techniques (ingest / transformation)

### 1. Structure-aware chunking — `rag_core/chunker.py`

`document_processor.py` currently destroys structure by flattening to text with `--- Page N ---`
string markers. Change it to return **structured blocks** (`{text, page, slide, heading, doc_type,
source}`) and let the chunker split *within* block boundaries so a chunk never straddles two pages.

- Child chunks: ~400 tokens (~1600 chars), 200-char overlap — the embedded/retrieved unit.
- Parent window: the containing page/slide/section, capped ~6000 chars — what actually reaches
  the LLM (**small-to-big retrieval**: precise matching, complete context).
- Heading detection for DOCX (paragraph styles) and PDF (line-shape heuristics) to populate
  `section` metadata.

### 2. Rich metadata per chunk — enables pre-filtering

Written at ingest, no LLM required: `source_file`, `doc_type` (pdf/docx/pptx), `page` / `slide`,
`section`, `chunk_index`, `char_start`/`char_end`, `word_count`, `parent_id`, `chunk_id`.

Surfaced two ways: **manual** sidebar filters (scope retrieval to chosen files/pages), and
**inferred** — the LLM converts "what did the Smith paper say in its methods?" into a Chroma
`where={"source_file": "smith2024.pdf", "section": {"$contains": "method"}}` clause applied
*before* the vector search. This is the metadata pre-filtering you asked about.

### 3. Optional LLM enrichment at ingest (default OFF)

Gated behind a sidebar checkbox because it costs one local inference per chunk:

- **Contextual chunk headers** — a 1-2 sentence LLM blurb situating each chunk in its document,
  prepended *before embedding*. Large retrieval-accuracy win, meaningful ingest cost.
- **Document-level summary + keywords** — one call per *document* (cheap), improves filter
  inference and gives the UI something to display.

### 4. Embeddings — `rag_core/embeddings.py`

Ollama-served, keeping the app 100% local: `nomic-embed-text` (768-dim, fast) by default,
`mxbai-embed-large` selectable. Batched with a `st.progress` bar.

> **Critical detail:** chromadb's *default* embedding function downloads an ONNX MiniLM model on
> first use. We must always pass our own embedding function explicitly so that download never
> happens and nothing but Ollama is ever contacted.

Startup check: `GET /api/tags` for an embedding-capable model; if absent, show
`ollama pull nomic-embed-text` and disable ingest rather than failing mid-run.

---

## Retrieval-side techniques

### 5. Hybrid retrieval — `rag_core/store.py` (required)

One `EphemeralRagStore` object held in `st.session_state`, wrapping:

- **Dense:** `chromadb.EphemeralClient()` — in-memory, nothing touches disk, dies with the process.
- **Sparse:** `rank_bm25.BM25Okapi` over the same chunk list, with light normalization
  (lowercase, punctuation strip, keep hyphenated tokens intact so gene names / statute numbers /
  acronyms survive).
- **Fusion:** Reciprocal Rank Fusion, `score = Σ 1/(k + rank)` with `k=60`. RRF over
  score-normalization because BM25 and cosine scores aren't comparable and RRF needs no tuning.

Defaults: dense top-20 + BM25 top-20 → RRF → top-12 → (optional re-rank) → top-5.
Sidebar exposes each `k` and a dense↔sparse weight slider.

`clear()` deletes the collection, drops the BM25 index and chunk list, and resets UI state — wired
to a **"🗑️ Clear Vector Store"** button. Process exit clears everything by construction.

### 6. Query transformation — `rag_core/query_transformer.py`

Each independently toggleable:

- **Conversational condensation** (default ON) — rewrites "how does that compare to the control?"
  into a standalone question using chat history. Without this, follow-up buttons retrieve garbage,
  since pronouns carry no embedding signal.
- **Query rewriting** (default ON) — restates the raw question in the vocabulary the documents
  likely use. This is precisely the "re-stating a more informed query" technique you described.
- **Multi-query fan-out** (default OFF) — 3 phrasings, retrieve each, RRF the union. Best recall
  gain per unit complexity; costs 1 LLM call + 3 embeddings.
- **HyDE** (default OFF) — embed a hypothetical *answer* rather than the question. Helps most on
  terse, jargon-heavy queries.
- **Filter inference** (default OFF) — emits the metadata `where` clause described in §2.

### 7. Re-ranking — `rag_core/reranker.py` (optional, as requested)

Over-retrieve then re-score. Two interchangeable backends behind one interface:

- **LLM batch re-ranker** (default when enabled) — all 12 candidates scored 0-10 in a *single*
  call with a forced-JSON response. No new dependency, ~1 extra inference.
- **Cross-encoder** (opt-in) — `sentence-transformers` + `ms-marco-MiniLM-L-6-v2`. Faster and
  better per-query, but a heavy dependency and a ~90 MB model download, so it stays optional.

Also **contextual compression** (default OFF): strip sentences from surviving chunks that don't
bear on the query, buying room in the context budget.

### 8. Assembly + citations — `rag_core/context_assembler.py`

- Expand winning child chunks to their parent windows, dedup overlaps, pack **to a token budget**
  (`num_ctx × 0.6`, char/4 heuristic) instead of blindly concatenating. **This is the direct fix
  for the silent-truncation bug.**
- Number sources `[S1]`, `[S2]`…, each mapped to real `(file, page/slide, section)` metadata.
- Prompt instructs inline `[S#]` citation; the UI renders an expandable **"📚 Sources"** panel per
  answer showing each retrieved chunk, its metadata, and its fused/re-rank scores.
- **Grounding check** (default OFF) — a verification pass flagging answer sentences carrying no
  `[S#]` support.

---

## Bugs fixed along the way

| Issue | Location | Fix |
|---|---|---|
| `num_ctx` ceiling 8192 vs 200 MB uploads | `app.py:263` | Raise max to 32768, default 8192; token-budgeted assembly makes overflow structurally impossible |
| Mode-prompt block duplicated verbatim | `app.py:495-530`, `app.py:585-615` | Single `rag_pipeline.answer()` call |
| `has_documents` accepted but never used | `prompt_builder.py:60-95` | Actually branch on it; move grounding constraint into the **system** prompt where it isn't diluted by history |
| History stores base query, loses evidence | `app.py:628` | Store `(query, response, source_ids)`; export includes sources |
| `python-pptx` missing despite PPTX support | `requirements.txt` | Add it |
| No OCR fallback for scanned PDFs | `document_processor.py:37` | Detect near-zero text yield and warn explicitly instead of silently returning `[No text could be extracted]` |
| `localhost:11434` hardcoded, no timeout | `app.py:53,281` | Module constant + timeout |

---

## Dependencies to add

```
chromadb>=0.5.0          # EphemeralClient — in-memory only
langchain-chroma>=0.1.0  # optional; direct chromadb client is enough
rank-bm25>=0.2.2         # sparse retrieval
python-pptx>=0.6.23      # ALREADY USED, currently missing from requirements
# optional, re-ranking only:
sentence-transformers>=3.0.0
```

No `tiktoken` — it doesn't match Ollama tokenizers anyway; a configurable chars/token heuristic is
honest and adequate for budgeting.

---

## Implementation phases

Each phase leaves the app working end to end.

| Phase | Delivers |
|---|---|
| **0** | Write this plan to repo root as `ADVANCED_RAG_PLAN.md`; add deps; fix `num_ctx`, `requirements.txt`, hardcoded URLs; scaffold `rag_core/` package; add `rag_core/embeddings.py` + startup model check |
| **1** | `document_processor` → structured blocks; `rag_core/chunker.py`; `rag_core/store.py` with Chroma-ephemeral ingest; "Clear Vector Store" button; chunk-inspector UI |
| **2** | BM25 + RRF hybrid retrieval; `rag_core/context_assembler.py`; `rag_core/pipeline.py`; route both `app.py` paths through it; Sources panel — **first phase where answer quality visibly improves** |
| **3** | `rag_core/query_transformer.py`: condensation + rewriting ON, multi-query/HyDE/filter-inference behind toggles |
| **4** | `rag_core/reranker.py`: LLM batch re-ranker + optional cross-encoder; contextual compression |
| **5** | Verified citations, grounding check, sidebar metadata filters, "Retrieval Diagnostics" expander showing what each stage did |
| **6** | Opt-in ingest enrichment (contextual chunk headers, doc summaries); small golden-question eval harness to measure whether toggles actually help |

---

## Verification

**Manual, end to end** (`streamlit run app.py`, Ollama running with a chat model and
`nomic-embed-text`):

1. **Ingest** — upload a multi-page PDF + a PPTX; confirm chunk count, page/slide metadata, and
   that ingest completes with a progress bar.
2. **Ephemerality** — click "Clear Vector Store" → retrieval returns nothing, STRICT mode says so
   rather than hallucinating. Restart the app → store is empty, nothing written to disk
   (`git status` clean; no `chroma*` files created).
3. **Hybrid proves itself** — query an exact rare token (a gene name, a figure number). BM25 must
   surface the chunk containing it; confirm via the Sources panel that it ranked highly even when
   dense similarity did not find it.
4. **Semantic proves itself** — query a paraphrase using none of the document's words; dense
   retrieval should still find the passage.
5. **The truncation bug is gone** — upload a 100+ page document with `num_ctx` at 2048. Ask about
   content on a late page. Pre-change this fails; post-change retrieval finds it and assembly stays
   within budget.
6. **Query transformation** — ask a question, then a pronoun-laden follow-up ("how does that
   compare to the control?"). Diagnostics must show the condensed standalone query, and retrieval
   must succeed where it previously would not.
7. **Re-ranking** — same query with the toggle off then on; compare the Sources ordering and note
   the latency delta.
8. **Grounding** — in STRICT mode ask something genuinely absent from the documents. Expected:
   an explicit "not found," no fabricated citation.
9. **Regressions** — templates CRUD, Perspective/Audience, follow-up buttons, and both HTML
   exports all still work; LLM mode (no documents) is unchanged.

**Automated:** unit tests (`rag_core` has no Streamlit dependency, so it is directly testable) for `chunker` (boundaries, overlap, metadata correctness), `store`
(RRF math against a hand-computed fixture; `clear()` empties every index), and `context_assembler`
(never exceeds budget; citation numbering is stable).

---

## Implementation status

All seven phases are implemented and verified. `rag_core/` carries no Streamlit
import and no persistence assumption, so it can be lifted into the persistent
chatbot app by swapping `EphemeralClient` for `PersistentClient` in
`rag_core/store.py`.

| Phase | Status | Landed as |
|---|---|---|
| 0 — Fixes & scaffolding | Done | `rag_core/{config,ollama_api,embeddings}.py`; `num_ctx` 256–32768 (default 8192); `python-pptx` added; embedding models split out of the chat dropdown |
| 1 — Ingest | Done | `chunker.py`, `store.py`; structured blocks from `document_processor.py`; Clear Vector Store button; chunk inspector |
| 2 — Hybrid retrieval | Done | BM25 + dense with RRF; `context_assembler.py`; `pipeline.py`; Sources panel; both duplicated prompt blocks removed |
| 3 — Query transformation | Done | `query_transformer.py` — condensation, rewriting, multi-query, HyDE, filter inference |
| 4 — Re-ranking | Done | `reranker.py` — LLM batch reranker + optional cross-encoder |
| 5 — Citations & diagnostics | Done | `citations.py`; Retrieval Diagnostics panel; sidebar metadata filters |
| 6 — Enrichment & eval | Done | `enrichment.py` (opt-in contextual headers); `tools/eval_retrieval.py` |

### Verified behaviour

Measured against generated fixtures with a live local model (`ornith:9b`,
`nomic-embed-text`):

- **Truncation bug fixed.** A fact on page 118 of a 120-page document was
  retrieved and answered correctly at `num_ctx=2048` — the setting under which
  the old code silently discarded 58% of the document.
- **Hybrid retrieval earns its place.** In `tools/eval_retrieval.py --demo` at
  top-k=1, keyword-only scores 67% and semantic/hybrid 83%; keyword-only misses
  both paraphrase questions.
- **Grounding holds.** Asked something absent from the documents in strict mode,
  the app answers "This information is not found in the provided documents."
  With an empty index, strict mode declines without calling the model at all.
- **Condensation works.** The bare follow-up "How does that compare to the
  control?" retrieves only the Controls page; condensed to a standalone
  question it retrieves both values and answers correctly.
- **Re-ranking discriminates.** On a 32-chunk corpus the target scored 10.0,
  the related control 5.0, and irrelevant log entries 0.0.
- **59 unit tests pass** (`pytest tests/ -q`), covering chunk boundaries, RRF
  against hand-computed values, budget enforcement, citation checking, sanitizer
  edge cases, and fail-soft behaviour in every LLM-backed stage.

### Deviations from the plan as approved

1. **Chat history keeps its `(query, response)` shape.** Sources and diagnostics
   are stored in parallel dicts keyed by turn index (the pattern already used
   for follow-up questions) rather than widening the tuple. Same outcome —
   history retains its evidence — without touching the HTML exporters.
2. **Two bugs found during implementation, both fixed.** `chromadb.EphemeralClient()`
   returns a process-wide singleton, so a second store collided on the collection
   name and would have crashed the app whenever the embedding model changed;
   collection names are now unique per instance and `close()` releases them.
   Separately, strict mode with an empty index fell through to answering from
   general knowledge, which is the exact substitution the mode exists to prevent.
3. **Mode banners now key off the index, not extracted text.** Previously the UI
   could promise "answers ONLY from your documents" while nothing was indexed.
4. **BM25 stopword filtering added.** Without it the sparse index ranked a
   naturally-phrased question on its grammar; an administrative passage
   outranked the correct one for "what happens when a cell decides to destroy
   itself".

### Known limitations

- **No OCR.** Scanned PDFs yield no text; this is now detected and reported at
  upload rather than surfacing later as an unexplained "not found".
- **Token budgeting is a 4-chars-per-token heuristic.** Ollama exposes no
  tokenizer per model; the estimate is deliberately conservative.
- **Enrichment costs one model call per chunk** and is repaid every session,
  since the index is ephemeral. Off by default, with an estimate shown first.
- **Re-ranking and query transformation each add a local model call**, which on
  this hardware is seconds per question. Both are off by default and reported in
  the diagnostics panel when active.
