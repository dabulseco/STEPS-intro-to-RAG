# 📦 Project Overview - Complete File Guide

## 🎯 What You Received

A complete, production-ready Streamlit application for interacting with local LLMs via Ollama, specifically designed for scientific and educational use. The application has been thoroughly refactored from your original code with significant new features added.

---

## 📁 File Structure & Purpose

```
STEPS-intro-to-RAG/                      📦 Main project directory
│
├── 🚀 QUICKSTART FILES (Start Here!)
│   ├── QUICKSTART.md                    ⭐ 5-minute setup guide - READ THIS FIRST
│   └── README.md                        📘 Comprehensive documentation
│
├── 🔧 APPLICATION FILES (Core Code)
│   ├── app.py                           💻 Streamlit UI -- all widgets live here
│   └── requirements.txt                 📋 Python dependencies
│
├── 📚 DOCUMENTATION FILES
│   ├── ARCHITECTURE.md                  🏗️  Visual system architecture
│   ├── CHANGELOG.md                     📝 Version history
│   ├── ADVANCED_RAG_PLAN.md             🔍 Retrieval engine design & rationale
│   └── TROUBLESHOOTING.md               🔧 Comprehensive problem-solving guide
│
├── ⚙️ CONFIG FILES (Template Storage)
│   └── config/
│       └── default_templates.json       🎯 Seed templates for "Reset to Defaults" (empty by default)
│       # question_templates.json        (Auto-created on first run)
│
├── 🔍 RETRIEVAL ENGINE (rag_core/)      ← No Streamlit; portable to other apps
│   ├── config.py                        🎛️  Tunable constants
│   ├── ollama_api.py                    🔌 Model discovery (chat vs embedding)
│   ├── embeddings.py                    🧬 Batched Ollama embedding client
│   ├── chunker.py                       ✂️  Structure-aware chunking + metadata
│   ├── store.py                         🗄️  Ephemeral Chroma + BM25 + RRF fusion
│   ├── query_transformer.py             🔄 Condense/rewrite/multi-query/HyDE
│   ├── reranker.py                      🏅 LLM & cross-encoder re-ranking
│   ├── context_assembler.py             📦 Token-budgeted packing + citations
│   ├── citations.py                     ✅ Citation verification
│   ├── enrichment.py                    🧠 Optional contextual chunk headers
│   └── pipeline.py                      🎯 Orchestrator -- the single answer path
│
├── 🧪 TESTS & TOOLS
│   ├── tests/test_rag_core.py           ✔️  60 unit tests (no Ollama needed)
│   └── tools/eval_retrieval.py          📏 Retrieval evaluation harness
│
└── 🛠️ UTILITY MODULES (App-specific)
    └── utils/
        ├── __init__.py                  📦 Package initialization
        ├── template_manager.py          🗂️  Template CRUD operations
        ├── prompt_builder.py            🎭 Dynamic prompt generation + grounding rules
        ├── html_export.py               📄 HTML export & Markdown
        └── document_processor.py        📑 PDF/DOCX/PPTX → structured blocks
```

### Why `rag_core/` is separate from `utils/`

`utils/` holds helpers specific to *this* app. `rag_core/` is a self-contained
retrieval engine that imports no Streamlit and assumes nothing about how long
the vector store lives. That boundary exists so the engine can be copied into
another application unchanged -- making it persistent is a one-line change in
`rag_core/store.py` (`EphemeralClient` → `PersistentClient`).

Streamlit is still the UI. All 379 widget calls live in `app.py`.

---

## 📖 Which Files to Read When

### 🏁 Getting Started (First 10 Minutes)
1. **QUICKSTART.md** - Quick setup and first steps
2. **README.md** - Features overview and installation
3. **requirements.txt** - See what you need to install

### 💻 Using the Application
1. **app.py** - Main application (run this!)
2. **QUICKSTART.md** - Usage examples
3. **README.md** - Detailed usage guide

### 🔨 Customizing & Extending
1. **config/default_templates.json** - See all templates
2. **utils/prompt_builder.py** - Modify perspectives/audiences
3. **utils/template_manager.py** - Understand template system
4. **ADVANCED_RAG_PLAN.md** - Retrieval engine design and extension points

### 🐛 Troubleshooting
1. **TROUBLESHOOTING.md** - Start here for any issues
2. **README.md** - Troubleshooting section
3. **CHANGELOG.md** - What changed from original

### 🏗️ Understanding Architecture
1. **ARCHITECTURE.md** - Visual diagrams and data flow
2. **ADVANCED_RAG_PLAN.md** - Design decisions and rationale
3. **CHANGELOG.md** - What's new vs original

---

## 🎯 File Details

### Core Application Files

#### `app.py` (468 lines)
**Purpose**: Main Streamlit application with UI and logic flow
**Key Sections**:
- Lines 1-50: Imports and configuration
- Lines 52-108: Helper functions (model fetch, AI response generation)
- Lines 110-162: Session state and navigation
- Lines 164-446: Chat page UI
- Lines 448-587: Template management page UI

**When to Edit**:
- Add new UI elements
- Change page layout
- Modify navigation
- Adjust sidebar organization

#### `requirements.txt` (6 lines)
**Purpose**: Python package dependencies
**Contents**:
```
streamlit>=1.32.0
requests>=2.31.0
langchain-ollama>=0.1.0
langchain-core>=0.1.0
markdown>=3.5.0
```

**When to Edit**:
- Add new Python packages
- Update version requirements
- Lock specific versions for production

### Configuration Files

#### `config/default_templates.json` (currently empty)
**Purpose**: Seed templates copied to the user file on first run, and restored
by "Reset to Defaults". **Ships empty**, so a fresh install starts with no
templates and "Reset to Defaults" clears the library rather than repopulating it.
Put templates here if you want a set that survives a reset.
**Structure**:
```json
{
  "templates": [
    {
      "id": "unique_id",
      "name": "Display Name",
      "template": "Question text",
      "category": "Category",
      "requires_input": true/false,
      "input_label": "Label for input field",
      "description": "What this does"
    }
  ]
}
```

**When to Edit**:
- Add more default templates
- Update existing template descriptions
- Reorganize categories

**Note**: Users shouldn't edit this directly - it's the master copy. Their changes go to `question_templates.json` (auto-created).

### Utility Modules

#### `utils/template_manager.py` (205 lines)
**Purpose**: Manages template CRUD operations and JSON storage
**Key Classes/Functions**:
- `TemplateManager`: Main class
  - `load_templates()`: Read from JSON
  - `save_templates()`: Write to JSON
  - `add_template()`: Create new template
  - `delete_template()`: Remove template
  - `get_template_by_id()`: Retrieve specific template
  - `get_templates_by_category()`: Filter by category
  - `reset_to_defaults()`: Restore original templates

**When to Edit**:
- Change storage format (e.g., to database)
- Add validation logic
- Implement template versioning
- Add template search features

#### `utils/prompt_builder.py` (178 lines)
**Purpose**: Constructs dynamic system prompts based on context
**Key Classes/Functions**:
- `PromptBuilder`: Main class
  - `PERSPECTIVES`: Dictionary of perspective definitions
  - `AUDIENCES`: Dictionary of audience definitions
  - `build_system_prompt()`: Generate contextual prompt
  - `_get_specific_guidance()`: Detailed guidance for combinations

**When to Edit**:
- Add new perspectives (add to `PERSPECTIVES` dict)
- Add new audiences (add to `AUDIENCES` dict)
- Customize guidance for specific combinations
- Change communication style directives

**Example Addition**:
```python
# Add a new perspective
PERSPECTIVES = {
    # ... existing ...
    "Industry Expert": {
        "description": "Professional with real-world experience",
        "style": "practical, solution-oriented, business-focused"
    }
}
```

#### `utils/html_export.py` (369 lines)
**Purpose**: HTML generation and Markdown conversion for exports
**Key Functions**:
- `build_pair_html()`: Single Q&A HTML document
- `build_conversation_html()`: Full conversation HTML
- `_markdown_to_html()`: Markdown to HTML conversion
- `_pipe_table_block_to_html()`: GitHub-style table conversion
- `safe_filename()`: Generate valid filenames

**When to Edit**:
- Change HTML styling (modify Bootstrap classes)
- Adjust export format
- Add PDF export capability
- Customize table rendering

### Retrieval Engine Modules (`rag_core/`)

None of these import Streamlit. Each can be tested and reused on its own.

#### `rag_core/pipeline.py` — the orchestrator
**Purpose**: The single answer path for the whole app. Both the chat input and
the template form call `answer()`, which is what keeps their grounding behaviour
identical (they used to carry separate copies of the prompt logic and could
drift apart).
**Key functions**: `answer()`, `retrieve()`, `resolve_mode()`
**Stages**: condense → transform → retrieve → rerank → assemble → generate

#### `rag_core/store.py` — hybrid index
**Purpose**: Holds two indexes over the same chunks — Chroma vectors for
semantic search, BM25 for keyword search — and fuses their rankings with
Reciprocal Rank Fusion.
**Key class**: `EphemeralRagStore` (`add_chunks`, `search`, `clear`, `close`)
**To make persistent**: change `_new_client()` to return
`chromadb.PersistentClient(path=...)`. That is the only line that decides it.

#### `rag_core/chunker.py` — splitting + metadata
**Purpose**: Turns extracted blocks into small "child" chunks for matching, each
carrying a larger "parent" window for the model to read, plus source/page/
section metadata. Chunks never straddle a page boundary.
**When to edit**: chunk sizes and overlap live in `rag_core/config.py`.

#### `rag_core/context_assembler.py` — the truncation fix
**Purpose**: Packs retrieved passages to a budget derived from `num_ctx` and
numbers them `[S1]`, `[S2]`. This is what makes silent prompt truncation
impossible; anything that will not fit is *reported*, not dropped quietly.

#### `rag_core/query_transformer.py` — query rewriting
**Purpose**: Condensation (resolving "that" against the conversation), query
rewriting, multi-query fan-out, HyDE, and metadata filter inference. Every stage
falls back to the user's original wording if the model returns something
unusable.

#### `rag_core/reranker.py` — precision pass
**Purpose**: Re-scores retrieved candidates against the question. `LLMReranker`
scores all candidates in one call; `CrossEncoderReranker` is more accurate but
needs `sentence-transformers`. Both fall back to retrieval order on failure.

#### `rag_core/citations.py` — citation checking
**Purpose**: Verifies every `[S#]` marker against the sources actually supplied,
so a fabricated citation is caught rather than trusted. Deterministic — no model
call.

#### `rag_core/embeddings.py`, `ollama_api.py`, `enrichment.py`, `config.py`
Batched embedding client; model discovery that separates chat from embedding
models; optional ingest-time contextual headers; and all tunable constants.

### Documentation Files

#### `README.md` (Comprehensive Guide)
**Sections**:
- Features overview
- Installation instructions
- Usage guide (basic workflow, template management)
- Use cases (scientists, educators, students)
- Customization guide
- Model parameters explained
- Troubleshooting
- Tips & best practices

**For**: Everyone (primary documentation)

#### `QUICKSTART.md` (Quick Guide)
**Sections**:
- 5-minute installation
- 2-minute first steps
- Key features to explore
- Tips for best results
- Common questions

**For**: New users wanting to get started fast

#### `ADVANCED_RAG_PLAN.md` (Retrieval Engine Design)
**Sections**:
- Why the retrieval layer was added, and the bug it fixes
- How ephemerality shapes the design
- Every technique used, ingest-side and retrieval-side
- Implementation status, verified behaviour, and known limitations

**For**: Developers extending retrieval, or porting `rag_core/` to another app

#### `ARCHITECTURE.md` (Visual Guide)
**Sections**:
- Application architecture diagram
- Data flow diagram
- Template system flow
- Component interaction diagrams

**For**: Understanding system design visually

#### `CHANGELOG.md` (Version History)
**Sections**:
- New features (what was added)
- Modified features (what changed)
- Preserved features (what stayed the same)
- Breaking changes (none!)
- Migration guide

**For**: Comparing with original code

#### `TROUBLESHOOTING.md` (Problem Solving)
**Sections**:
- Quick diagnostics checklist
- 10 common issues with solutions
- Advanced troubleshooting
- Platform-specific issues
- Best practices

**For**: When things don't work as expected

---

## 🎓 Development Workflow

### For Scientists (Non-Programmers)

**Just Using the App**:
1. Read: `QUICKSTART.md`
2. Run: `streamlit run app.py`
3. Reference: `README.md` for features
4. If stuck: `TROUBLESHOOTING.md`

**Adding Custom Templates** (No coding required):
1. Run the app
2. Navigate to "Manage Templates"
3. Use "Add Template" tab
4. Fill in the form
5. Save and use immediately

**Customizing Perspectives** (Minimal coding):
1. Open: `utils/prompt_builder.py`
2. Find: `PERSPECTIVES` or `AUDIENCES` dictionary
3. Add your entry following the existing pattern
4. Save and restart app

### For Developers

**Understanding the Code**:
1. Read: `ARCHITECTURE.md` - See system design
2. Read: `ADVANCED_RAG_PLAN.md` - Understand design decisions
3. Review: `app.py` - Main flow
4. Review: `utils/` files - Business logic

**Making Changes**:
1. **UI Changes**: Edit `app.py`
2. **Template Logic**: Edit `utils/template_manager.py`
3. **Prompt Logic**: Edit `utils/prompt_builder.py`
4. **Export Format**: Edit `utils/html_export.py`
5. **Add Features**: Follow patterns in `app.py`

**Testing Changes**:
```bash
# Run with debug logging
streamlit run app.py --logger.level debug

# Test specific module
python3 -c "from utils.template_manager import TemplateManager; tm = TemplateManager(); print(len(tm.load_templates()))"
```

---

## 💾 Important Files to Backup

### Must Backup (Contains Your Data)
- `config/question_templates.json` - Your custom templates

### Good to Backup (Easy to Restore)
- `config/default_templates.json` - Default templates (rarely changes)
- All `.md` documentation files

### Don't Need to Backup (Generated or Easily Restored)
- `__pycache__/` directories
- `.pyc` files
- Streamlit cache files

**Backup Command**:
```bash
# Backup important files
cp config/question_templates.json config/question_templates_backup_$(date +%Y%m%d).json

# Or backup entire config directory
cp -r config config_backup_$(date +%Y%m%d)
```

---

## 🔍 Quick Reference

### File Size Reference
```
app.py                          ~25 KB
utils/html_export.py           ~19 KB
utils/template_manager.py      ~11 KB
utils/prompt_builder.py        ~10 KB
config/default_templates.json  ~12 KB
README.md                      ~25 KB
ARCHITECTURE.md                ~10 KB
CHANGELOG.md                   ~12 KB
TROUBLESHOOTING.md            ~15 KB
QUICKSTART.md                  ~8 KB
```

### Line Count Reference
```
Python Code:
  app.py:                 468 lines
  template_manager.py:    205 lines
  prompt_builder.py:      178 lines
  html_export.py:         369 lines
  Total Code:           1,220 lines

Documentation:
  All .md files:        ~1,800 lines
  Total Project:        ~3,000 lines
```

---

## 🚀 Next Steps

1. **Read**: `QUICKSTART.md` (5 minutes)
2. **Install**: Follow installation steps (5 minutes)
3. **Run**: `streamlit run app.py`
4. **Explore**: Try different templates and settings
5. **Customize**: Add your own templates
6. **Reference**: Keep `README.md` handy

---

## 🎉 You're All Set!

You have everything you need:
- ✅ Complete working application
- ✅ 30 ready-to-use templates
- ✅ Comprehensive documentation
- ✅ Troubleshooting guide
- ✅ Extension guides
- ✅ Visual architecture diagrams

**Start with**: `QUICKSTART.md`

**Happy exploring! 🔬**
