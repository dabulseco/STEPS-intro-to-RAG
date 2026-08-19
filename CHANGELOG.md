# Changelog: v1.0 → v2.0

## Overview
This document tracks all changes from the original `db9d_LLMMemoryParams.py` to the new refactored application.

---

## 🆕 NEW FEATURES

### 1. Perspective & Audience System
**Status**: ✅ Fully Implemented

**What's New**:
- Dropdown selector for "Perspective" (who's writing the answer)
  - Options: Scientist, High School Teacher, Lay Person
- Dropdown selector for "Audience" (who the answer is for)
  - Options: Scientist, College Student, High School Student, Middle School Student, High School Teacher, Lay Person
- Dynamic system prompt generation based on selected combination
- 18 unique perspective-audience combinations with tailored guidance
- Settings visible in sidebar, apply to entire conversation
- Context indicators in HTML exports

**Files**:
- `utils/prompt_builder.py` (NEW)
- `app.py` (modified - sidebar section added)

### 2. Question Template System
**Status**: ✅ Fully Implemented

**What's New**:
- 30 pre-built templates covering:
  - Research (5 templates)
  - Teaching (7 templates)
  - Laboratory (4 templates)
  - Research Design (3 templates)
  - Statistics (3 templates)
  - Critical Thinking (2 templates)
  - Career (1 template)
  - Current Science (1 template)
  - General (4 templates)
- Category-based organization and filtering
- Two usage modes:
  - Form mode with structured input
  - Quick chat input population
- Template browser in expandable section
- Persistent storage in JSON format

**Files**:
- `config/default_templates.json` (NEW - 30 templates)
- `config/question_templates.json` (NEW - user's templates, auto-created)
- `utils/template_manager.py` (NEW)
- `app.py` (modified - template section added)

### 3. Template Management Interface
**Status**: ✅ Fully Implemented

**What's New**:
- Separate "Manage Templates" page (accessible via sidebar navigation)
- Three tabs:
  - **View Templates**: Browse, filter, view details, delete
  - **Add Template**: Create new custom templates with form
  - **Settings**: Statistics display, reset to defaults
- Full CRUD operations (Create, Read, Update, Delete)
- Category management
- Template statistics (total count, category count)

**Files**:
- `app.py` (modified - new page added)
- `utils/template_manager.py` (NEW)

---

## 📝 MODIFIED FEATURES

### 1. Application Structure
**Before**: Single file with all code (~400 lines)
**After**: Modular structure with separate files (~1,220 lines total)

**Changes**:
- Extracted HTML functions → `utils/html_export.py`
- Created template management → `utils/template_manager.py`
- Created prompt building → `utils/prompt_builder.py`
- Main app focused on UI flow only

### 2. HTML Export Functions
**Before**: Inline functions in main file
**After**: Separate module with enhanced features

**Changes**:
- Moved to `utils/html_export.py`
- Added context indicators (perspective/audience) to exports
- Enhanced docstrings
- Better error handling
- All original functionality preserved

**Files**:
- `utils/html_export.py` (NEW - extracted from original)
- `app.py` (modified - imports from utils)

### 3. User Interface
**Before**: Single page with all controls
**After**: Multi-page with organized sections

**Changes**:
- Added navigation system (Chat vs Manage Templates)
- Reorganized sidebar:
  - Navigation at top
  - Context settings (NEW)
  - Model settings (preserved)
  - Question history (preserved)
- Template browser in expandable section (NEW)
- Clearer visual hierarchy

**Files**:
- `app.py` (modified - UI restructure)

### 4. System Prompt
**Before**: Static system prompt
**After**: Dynamic, context-aware prompt

**Changes**:
- Now generated based on perspective and audience
- Includes specific guidance for each combination
- Maintains core principles (accuracy, conciseness)
- Enhanced with communication style directives

**Files**:
- `app.py` (modified - calls PromptBuilder)
- `utils/prompt_builder.py` (NEW)

### 5. Conversation History Display
**Before**: Simple markdown display
**After**: Enhanced with anchor navigation and better formatting

**Changes**:
- Added HTML anchors for sidebar links
- Better visual separation between Q&A pairs
- Context display (perspective/audience)
- Improved button placement

**Files**:
- `app.py` (modified - display section)

---

## 🔄 PRESERVED FEATURES

### ✅ All Original Functionality Maintained

1. **Model Selection**
   - Auto-detection of Ollama models
   - Dropdown selector
   - ✅ No changes

2. **Model Parameters**
   - Temperature slider (0.0 - 2.0)
   - Top-K slider (1 - 100)
   - Top-P slider (0.0 - 1.0)
   - Context length slider (256 - 8192)
   - ✅ All preserved, same defaults

3. **Chat Functionality**
   - Chat input
   - Message submission
   - Response generation
   - Conversation memory (last 5 exchanges)
   - ✅ All preserved

4. **Conversation History**
   - Full history display
   - Question numbering
   - ✅ Enhanced but preserved

5. **HTML Export**
   - Individual Q&A exports
   - Full conversation exports
   - Bootstrap styling
   - Markdown table support
   - ✅ Enhanced with context info, core preserved

6. **Sidebar Navigation**
   - Questions asked list
   - Links to specific questions
   - ✅ Preserved and enhanced

7. **Session State Management**
   - Chat history persistence
   - State across reruns
   - ✅ Preserved, expanded for new features

---

## 📦 NEW FILES

```
ollama_app/
├── app.py                          ✅ Refactored from db9d_LLMMemoryParams.py
├── requirements.txt                 ✅ Updated
├── README.md                        ✅ Comprehensive new documentation
├── QUICKSTART.md                   ✅ NEW - Quick start guide
├── IMPLEMENTATION_SUMMARY.md       ✅ NEW - Implementation details
├── ARCHITECTURE.md                 ✅ NEW - Visual architecture
├── CHANGELOG.md                    ✅ NEW - This file
├── config/
│   ├── default_templates.json      ✅ NEW - 30 default templates
│   └── question_templates.json     ✅ NEW - Auto-created user templates
└── utils/
    ├── __init__.py                 ✅ NEW - Package initialization
    ├── template_manager.py         ✅ NEW - Template operations
    ├── prompt_builder.py           ✅ NEW - Dynamic prompts
    └── html_export.py              ✅ NEW - Extracted from original
```

---

## 🔧 BREAKING CHANGES

**None!** All original functionality is preserved. The refactored code:
- Maintains all original features
- Uses same dependencies (plus no new required ones)
- Works with same Ollama setup
- Produces same quality responses
- Exports same HTML format (enhanced but compatible)

---

## 📊 Code Statistics

### Line Count Comparison

| Component | Original | New | Change |
|-----------|----------|-----|--------|
| Main App | ~400 lines | 468 lines | +68 (UI enhancements) |
| HTML Export | (inline) | 369 lines | Extracted |
| Template Manager | - | 205 lines | NEW |
| Prompt Builder | - | 178 lines | NEW |
| **Total** | ~400 lines | ~1,220 lines | +820 lines |

### Feature Count

| Feature Category | Original | New | Added |
|-----------------|----------|-----|-------|
| Core Features | 5 | 5 | 0 (preserved) |
| New Features | 0 | 3 | +3 |
| Configuration Options | 4 | 6 | +2 |
| UI Pages | 1 | 2 | +1 |
| Templates | 0 | 30 | +30 |

---

## 🎯 Implementation Goals - Status

| Goal | Status | Details |
|------|--------|---------|
| Add Perspective selector | ✅ Complete | 3 perspectives available |
| Add Audience selector | ✅ Complete | 6 audiences available |
| Create question templates | ✅ Complete | 30 comprehensive templates |
| Template menu/dropdown | ✅ Complete | Category-filtered browser |
| Store templates (JSON) | ✅ Complete | Persistent storage system |
| Retrieve templates | ✅ Complete | Load/save/search functions |
| Add custom templates | ✅ Complete | Full UI for adding |
| Delete templates | ✅ Complete | UI with confirmation |
| Two template modes | ✅ Complete | Form + Chat input |
| Apply to conversation | ✅ Complete | Settings apply globally |
| Template management UI | ✅ Complete | Separate page with tabs |
| Documentation | ✅ Complete | 4 documentation files |
| Maintain existing features | ✅ Complete | All preserved |
| Clean code structure | ✅ Complete | Modular, well-documented |

---

## 🚀 Migration Guide

### From Original to New Version

**Step 1**: No migration needed! The new version is a complete replacement.

**Step 2**: Install (same as before):
```bash
pip install -r requirements.txt
```

**Step 3**: Run (same command):
```bash
streamlit run app.py
```

**Your Data**: 
- Old exports remain compatible (HTML format unchanged)
- No database or state migration needed
- Fresh start with new features immediately available

---

## 🔮 Future Enhancements (Not in v2.0)

Potential features for future versions:
- Template import/export (share templates with others)
- Template usage analytics (track most-used)
- Template variables with validation
- Multi-language support
- Batch processing
- API integration for external tools
- Template categories customization
- Voice input/output
- Image/document upload

---

## 📞 Support & Feedback

For questions about changes:
- Review `ARCHITECTURE.md` for system design and `FILE_GUIDE.md` for the file map
- Check `README.md` for usage instructions
- See `QUICKSTART.md` for getting started
- Review `ARCHITECTURE.md` for technical details

---

**Version**: 2.0  
**Release Date**: February 2026  
**Upgrade**: Recommended for all users (no breaking changes)


---

# Changelog: v3.0 → v4.0 (Advanced Retrieval)

**Release Date**: August 2026

## Overview

v4.0 replaces prompt-stuffing with a real retrieval pipeline. The user-facing
modes (STRICT RAG / HYBRID / LLM) behave as before; what changed is that
documents are now *searched* rather than pasted wholesale into the prompt.

## The problem this fixes

`num_ctx` capped at 8,192 tokens and defaulted to 2,048, while uploads were
allowed up to 200 MB per file. Every uploaded document was concatenated and sent
with every question, so Ollama silently truncated it — on a 20-page paper the
model never saw most of the text, then answered confidently from the fragment
that survived, or reported "not found in the provided document". This read as
"the model is bad" when the pipeline was at fault.

## Added

### Retrieval engine (`rag_core/`, new package)
- **Hybrid search**: Chroma vectors (semantic) + BM25 (keyword), fused with
  Reciprocal Rank Fusion. Neither alone suffices — dense retrieval misses exact
  identifiers, sparse misses paraphrases.
- **Ephemeral vector store**: in-memory ChromaDB. Nothing is written to disk;
  cleared on demand or when the app closes.
- **Structure-aware chunking**: chunks never straddle a page or slide, and carry
  source, page/slide, section and character span.
- **Small-to-big retrieval**: precise child chunks are matched; the surrounding
  parent passage is what reaches the model.
- **Token-budgeted assembly**: context is packed to a budget derived from
  `num_ctx`. Overflow is now structurally impossible, and anything dropped is
  reported rather than lost silently.
- **Query transformation**: follow-up condensation (on by default), query
  rewriting, multi-query fan-out, HyDE, and metadata filter inference.
- **Re-ranking**: LLM batch reranker (no extra dependency) or cross-encoder
  (optional `sentence-transformers`).
- **Verified citations**: answers cite `[S1]`-style markers; every marker is
  checked against the sources actually supplied, and fabrications are flagged.
- **Optional ingest enrichment**: contextual chunk headers and document
  summaries, off by default with a cost estimate shown first.

### UI (Streamlit, in `app.py`)
- Embedding model selector, separate from the chat model
- Vector store panel with chunk counts and a **Clear Vector Store** button
- **🔍 Inspect Indexed Chunks** — see exactly what retrieval searches over
- **📚 Sources** panel per answer, with page/section and retrieval scores
- **🔬 Retrieval Diagnostics** panel — what was searched, found, and budgeted
- **🎛️ Retrieval Tuning** — semantic↔keyword balance, passage counts, document filter
- **🧪 Advanced Techniques** — per-technique toggles with latency warnings

### Tests and tooling
- `tests/test_rag_core.py` — 60 unit tests, no Ollama required (~0.3s)
- `tools/eval_retrieval.py` — retrieval evaluation harness reporting hit rate
  and MRR per configuration, so setting changes can be measured rather than guessed

## Changed

- **`num_ctx` range**: 256–8,192 (default 2,048) → 256–32,768 (default 8,192)
- **Model dropdown** no longer lists embedding models, which cannot generate text
- **Grounding rules moved to the system prompt**, where they hold for the whole
  exchange instead of weakening as chat history grows
- **Document extraction returns structured blocks** carrying page/slide and
  section, rather than one flat string
- **Mode banners key off the index**, not extracted text
- **Both answer paths unified**: chat input and template form now share one
  function, so their grounding cannot drift apart

## Fixed

- **Silent prompt truncation** — the headline bug above. Verified: a fact on
  page 118 of a 120-page document is now retrieved correctly at `num_ctx=2048`,
  the setting under which the old code discarded 58% of the document.
- **Strict mode fell through to general knowledge** when nothing was indexed —
  the exact substitution the mode exists to prevent. It now declines outright.
- **Duplicated prompt-assembly blocks** (~70 lines copy-pasted between the chat
  and template paths) collapsed into one call.
- **`has_documents` parameter accepted but never used** in `build_system_prompt`.
- **Second vector store crashed the app**: `chromadb.EphemeralClient()` returns a
  process-wide singleton, so switching embedding model collided on the collection
  name. Collection names are now unique per instance.
- **Mode banner could claim "answers ONLY from your documents"** while nothing
  was indexed.
- **BM25 ranked on grammar**: stopword filtering added after an administrative
  passage outranked the correct one for a naturally-phrased question.
- **`python-pptx` missing from `requirements.txt`** despite PPTX being supported.
- **Scanned PDFs failed silently** — now detected and reported at upload.
- **Duplicate filenames collided** in chunk IDs.

## Migration

No breaking changes to templates, exports, or saved data. Two steps:

```bash
pip install -r requirements.txt     # adds chromadb, rank-bm25, python-pptx
ollama pull nomic-embed-text        # required for document search
```

Without an embedding model the app still runs, but documents cannot be indexed —
and it says so plainly rather than answering from general knowledge while
appearing to read your files.

## Known limitations

- **No OCR.** Scanned PDFs yield no text; detected and reported at upload.
- **Token budgeting is a 4-chars-per-token heuristic** — Ollama exposes no
  per-model tokenizer. Deliberately conservative.
- **Enrichment costs one model call per chunk**, repaid every session since the
  index is ephemeral. Off by default.
- **Advanced techniques each add a local model call.** With all enabled a query
  took ~40s versus ~12s baseline on the test hardware.

---

**Version**: 4.0
**Release Date**: August 2026
**Upgrade**: Recommended. No breaking changes; requires an embedding model for
document search.
