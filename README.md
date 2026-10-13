# 🔬 Local LLM Scientific Assistant

A project of the [Institute for Future Intelligence](https://intofuture.org), built for the AI-STEPS faculty development program at Windward Community College and Kapiʻolani Community College.

A comprehensive Streamlit application for interacting with local Large Language Models (LLMs) via Ollama, designed specifically for scientific and educational use cases.

## ✨ Features

### 🔍 Advanced Retrieval (RAG)
- **Real retrieval pipeline**: documents are chunked, embedded, and searched — not pasted wholesale into the prompt
- **Hybrid search**: semantic (vector) + keyword (BM25) retrieval fused with Reciprocal Rank Fusion, so both paraphrases and exact identifiers (gene names, catalog numbers) are findable
- **Ephemeral vector store**: in-memory ChromaDB, cleared on demand or when the app closes — nothing is written to disk
- **Structure-aware chunking**: chunks never straddle a page or slide, and each carries its source, page/slide, and section
- **Small-to-big retrieval**: precise child chunks are matched, but the surrounding passage is what reaches the model
- **Token-budgeted context**: retrieved evidence is packed to fit the context window instead of being silently truncated
- **Verified citations**: answers cite `[S1]`-style markers traced to real passages, and fabricated markers are flagged automatically
- **Query transformation**: follow-up condensation, query rewriting, multi-query fan-out, and HyDE — each independently toggleable
- **Re-ranking**: optional LLM or cross-encoder re-scoring of retrieved candidates
- **Metadata pre-filtering**: restrict search to chosen documents, manually or inferred from the question
- **Retrieval diagnostics**: per-answer panel showing what was searched, what was found, and how the context budget was spent

### 📄 Document Upload & Analysis
- **Upload PDF, DOCX, and PPTX files**: Extract text from research papers, reports, documents, and presentations
- **Multiple file support**: Upload and analyze multiple documents simultaneously
- **200 MB per file limit**: Support for large documents and presentation files
- **Automatic retrieval**: Relevant passages are found and cited for every question
- **Preview capability**: View extracted text and inspect indexed chunks before querying

### 🎭 Context-Aware Responses
- **Perspective Setting**: Choose who's writing the response (Scientist, High School Teacher, Lay Person)
- **Audience Targeting**: Tailor responses for specific audiences (Scientists, College Students, High School Students, Middle School Students, Teachers, Lay Persons)
- Settings apply to entire conversation for consistent communication style

### 📋 Question Template System
- **Build your own library**: Save questions you ask repeatedly and reuse them in one click
- **Organized by Category**: Research, Teaching, Laboratory, Statistics, Critical Thinking, and more
- **Two Usage Modes**:
  - **Form Mode**: Structured input with dedicated fields for complex queries
  - **Chat Mode**: Quick template insertion into chat input
- **Template Management**: Full CRUD operations - Create, View, Delete templates
- **Persistent Storage**: Templates stored in JSON format for easy backup and sharing

> **The app ships with no pre-built templates.** `config/default_templates.json`
> is intentionally empty, so a fresh install starts with an empty library and you
> add your own from **Manage Templates → Add Template**.
>
> ⚠️ Because the defaults file is empty, **"Reset to Defaults" deletes every
> template you have created and restores nothing.** Back up
> `config/question_templates.json` before using it. To make it useful, copy your
> templates into `config/default_templates.json` — that file is what "reset"
> restores from.

### 💡 Suggested Follow-up Questions
- **Automatic generation**: After every AI response, the app generates 3 contextually relevant follow-up questions
- **One-click submission**: Click any suggested question to instantly submit it as your next query
- **Full pipeline support**: Clicked follow-up questions go through the same RAG/Hybrid/LLM mode logic as manually typed queries
- **Per-exchange storage**: Each Q&A pair retains its own set of follow-up suggestions throughout the session

### 💬 Advanced Chat Interface
- **Conversation History**: Full chat history with context retention (last 5 exchanges)
- **Export Capabilities**:
  - Individual Q&A pairs as standalone HTML documents
  - Full conversation transcripts with navigation
  - Bootstrap-styled exports with Markdown table support
- **Context Indicators**: Exported documents show perspective and audience settings

### ⚙️ Model Configuration
- **Dynamic Model Selection**: Automatically detects available Ollama models
- **Fine-tuned Parameters**:
  - Temperature control (0.0 - 2.0)
  - Top-K sampling (1 - 100)
  - Top-P (nucleus) sampling (0.0 - 1.0)
  - Context window size (256 - 32,768 tokens)
- **Separate embedding model selection**: embedding models are detected and listed separately from chat models

## 📚 Documentation Website

A browsable guide written for non-specialists — faculty, students, and anyone who
needs to understand what a RAG system is doing before relying on it — lives in
[`docs/`](docs/):

```bash
open docs/index.html          # macOS
# or serve it:  cd docs && python3 -m http.server 8000
```

It documents **every** sidebar control with its range, default and effect; each
advanced retrieval technique with the problem it solves and what it costs;
recommended model settings with three presets; and a plain-language glossary.
The folder is named `docs/` so it can be published free via GitHub Pages
(Settings → Pages → Deploy from branch → `/docs`).

## 🗂️ Persistent Library App

A second, separate app — `library_app.py` — keeps documents **between sessions**.
The main app (`app.py`) is unchanged and still forgets everything when closed.

```bash
python -m streamlit run library_app.py
```

- **Libraries**: named collections of documents, stored on your computer. Reopen
  one and ask questions immediately — no re-uploading or re-embedding.
- **Duplicate detection**: re-uploading a file (or the same text under another
  name) is skipped; the same DOI or title + year asks whether to skip, replace or
  keep both.
- **Publication details**: title, authors, year, journal and DOI per document,
  read from the DOI in the text, your chat model and the file's properties — all
  offline. Editable if wrong.
- **A chatbot per library**: chat model, perspective/audience, retrieval settings
  and custom instructions are saved with the library. Re-ranking is on by default.
- **Saved conversations**: listed in the sidebar; resume, rename, delete or export
  to HTML. Each answer keeps copies of the passages it cited.
- **Not included**: answer evaluation and question templates (use the main app).

**Storage.** Each library is a folder in `libraries/` (set `RAG_LIBRARY_DIR` to
keep them elsewhere). The folder holds the documents' extracted **text** and your
conversations, but not the original files. Back a library up by copying its
folder while the app is closed. `libraries/` is git-ignored so it can never be
pushed by accident. A library can be open in only one running copy of the app.

Sized for roughly 50–500 documents per library. Full guide:
[`docs/libraries.html`](docs/libraries.html).

## 📁 Project Structure

```
STEPS-intro-to-RAG/
├── app.py                          # Streamlit UI (view layer only)
├── library_app.py                  # Persistent Library App (saved libraries + chats)
├── library_ui/                     # Library App screens: chat, documents, settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── ADVANCED_RAG_PLAN.md            # Design and rationale for the retrieval engine
├── docs/                           # Documentation website (Bootstrap 5, no build step)
│   ├── index.html … glossary.html  # Concepts, controls, techniques, settings, evaluation, libraries
│   └── assets/                     # Stylesheet + navigation/filtering script
├── config/
│   ├── default_templates.json      # Seed templates restored by "Reset to Defaults" (empty by default)
│   └── question_templates.json     # User's custom templates (auto-created)
├── rag_core/                       # Retrieval engine -- no Streamlit, no persistence assumptions
│   ├── config.py                   # Tunable constants
│   ├── ollama_api.py               # Model discovery (chat vs embedding)
│   ├── embeddings.py               # Batched Ollama embedding client
│   ├── chunker.py                  # Structure-aware chunking + metadata
│   ├── store.py                    # Ephemeral Chroma + BM25 + RRF fusion
│   ├── query_transformer.py        # Condense / rewrite / multi-query / HyDE / filter inference
│   ├── reranker.py                 # LLM and cross-encoder re-ranking
│   ├── context_assembler.py        # Token-budgeted context packing + citations
│   ├── citations.py                # Citation verification
│   ├── enrichment.py               # Optional ingest-time contextual headers
│   ├── pipeline.py                 # Orchestrator -- the single answer path
│   ├── eval.py, evaluator_prompts.py # LLM-as-judge answer evaluation (main app)
│   ├── persistent_store.py         # On-disk Chroma store (Library App)
│   ├── library_db.py               # SQLite: documents, chunks (Library App)
│   ├── library.py, library_lock.py # Libraries: create/open, duplicates, recovery, re-embed
│   ├── doc_metadata.py             # Fingerprints, DOI, publication details
│   ├── conversations.py            # Saved conversations
│   └── chat_session.py             # Answering against a library
├── tests/
│   ├── test_rag_core.py            # Unit tests (no Ollama required)
│   └── test_library*.py, …         # Library App tests (no Ollama required)
├── tools/
│   └── eval_retrieval.py           # Retrieval evaluation harness
└── utils/
    ├── template_manager.py         # Template CRUD operations
    ├── prompt_builder.py           # Dynamic system prompt generation
    ├── html_export.py              # HTML export with Markdown support
    └── document_processor.py       # PDF, DOCX, and PPTX extraction into structured blocks
```

`rag_core/` imports no Streamlit and makes no assumption about how long the
vector store lives: the main app uses `EphemeralRagStore`, the Library App
`PersistentRagStore`, and both go through the same `pipeline.py`.

## 🚀 Installation

### Prerequisites
1. **Ollama**: Install from [ollama.ai](https://ollama.ai)
2. **Python**: Version 3.8 or higher

### Setup Steps

1. **Clone or download this repository**
```bash
git clone https://github.com/dabulseco/STEPS-intro-to-RAG.git
cd STEPS-intro-to-RAG
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (UI framework)
- LangChain (LLM integration)
- PyPDF2 (PDF text extraction)
- python-docx (DOCX text extraction)
- python-pptx (PPTX text extraction)
- chromadb (in-memory vector store)
- rank-bm25 (keyword retrieval)
- Other supporting libraries

3. **Start Ollama** (in a separate terminal)
```bash
ollama serve
```

4. **Pull an embedding model** (required for document retrieval)
```bash
ollama pull nomic-embed-text
```
Without one, documents can still be uploaded and read, but they cannot be
indexed or searched — and the app will tell you so rather than pretending
otherwise.

5. **Pull at least one chat model** (if you haven't already)
```bash
ollama pull llama2
# or
ollama pull mistral
# or any other model you prefer
```

6. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Basic Workflow

1. **Set Context** (Sidebar):
   - Choose your **Perspective** (who's writing)
   - Choose your **Audience** (who's reading)
   - Select your preferred **LLM model**
   - Adjust model parameters if needed

2. **Ask Questions** (Main Page - Chat):
   - **Option A - Upload Documents**:
     - Click "📄 Upload Documents" expander
     - Upload PDF, DOCX, or PPTX files (up to 200 MB each)
     - Click "Process Documents"
     - Documents are automatically included in all queries
   - **Option B - Free Form**: Type directly in the chat input
   - **Option C - Use Template**: 
     - Click "📋 Question Templates" expander
     - Browse by category
     - Select a template
     - Choose "Use Template with Form" for structured input, or
     - Choose "Populate Chat Input" for quick use

3. **Review & Explore**:
   - View conversation history with full responses
   - Click any of the **💡 Suggested Follow-up Questions** displayed after each answer to instantly deepen the conversation
   - Download individual Q&A pairs
   - Download full conversation transcript
   - All exports include your perspective/audience settings

### Template Management

Navigate to **Manage Templates** page (sidebar navigation):

#### Viewing Templates
- Browse all templates organized by category
- View template details (ID, category, description, template text)
- Delete templates you don't need

#### Adding Custom Templates
1. Click "➕ Add Template" tab
2. Fill in the form:
   - **Name**: Descriptive name for your template
   - **Category**: Choose from existing categories or use "Custom"
   - **Template Text**: The actual question/prompt
   - **Description**: What this template does (optional but recommended)
   - **Requires Input**: Check if template needs user content
   - **Input Label**: Custom label for input field
3. Click "Add Template"

#### Settings
- View statistics (total templates, categories)
- Reset to default templates if needed (⚠️ deletes custom templates)

## 🎯 Use Cases

### For Scientists
- **Literature Review**: "Summarize Research Paper" template
- **Method Analysis**: "Explain Research Methods" template
- **Experimental Design**: Design studies with appropriate controls
- **Data Analysis**: Statistical guidance and interpretation

### For Educators
- **Lesson Planning**: Complete lesson plan generation
- **Concept Explanation**: Multiple difficulty levels via audience setting
- **Creating Analogies**: Make complex topics accessible
- **Study Materials**: Generate study guides and review questions

### For Students
- **Homework Help**: Explain concepts at appropriate level
- **Exam Preparation**: Create practice questions and study guides
- **Research Projects**: Hypothesis development and experimental design
- **Lab Reports**: Protocol design and troubleshooting

## 🔧 Customization

### Adding Your Own Templates

Templates are stored in JSON format. You can:

1. **Use the UI** (recommended): Add templates via "Manage Templates" page
2. **Edit JSON directly**: Modify `config/question_templates.json`

**JSON Structure:**
```json
{
  "templates": [
    {
      "id": "unique_id",
      "name": "Display Name",
      "template": "Your question text here",
      "category": "Research",
      "requires_input": true,
      "input_label": "Enter your content:",
      "description": "What this template does"
    }
  ]
}
```

### Modifying Perspectives & Audiences

Edit `utils/prompt_builder.py`:
- Add new perspectives to `PERSPECTIVES` dictionary
- Add new audiences to `AUDIENCES` dictionary
- Customize guidance for specific combinations in `_get_specific_guidance()`

## 📊 Model Parameters Explained

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| **Temperature** | 0.0 - 2.0 | 0.3 | Controls randomness: lower = focused, higher = creative |
| **Top-K** | 1 - 100 | 40 | Number of highest probability tokens to consider |
| **Top-P** | 0.0 - 1.0 | 0.9 | Nucleus sampling: cumulative probability threshold |
| **Context Length** | 256 - 8192 | 2048 | Maximum conversation history size in tokens |

### Recommended Settings

- **Factual Questions**: Temperature 0.1-0.3, Top-K 20-40
- **Creative Writing**: Temperature 0.7-1.2, Top-K 60-80
- **General Use**: Default settings work well

## 🐛 Troubleshooting

### "No models available" Error
- Ensure Ollama is running: `ollama serve`
- Check if models are installed: `ollama list`
- Try pulling a model: `ollama pull llama2`

### Templates Not Loading
- Check if `config/default_templates.json` exists
- Verify JSON syntax if you edited manually
- Use "Reset to Defaults" in Template Settings

### Slow Response Times
- Reduce context length (num_ctx)
- Use smaller models (e.g., llama2:7b instead of llama2:13b)
- Reduce conversation history (currently limited to last 5 exchanges)

### Export Issues
- Ensure you have write permissions in the directory
- Check browser's download settings
- Try a different browser if downloads fail

## 💡 Tips & Best Practices

1. **Context Settings**: Set perspective and audience BEFORE starting conversation for consistent responses
2. **Follow-up Questions**: Use the suggested follow-up questions to go deeper on any topic without having to type — simply click them
3. **Template Library**: Review all default templates to discover useful patterns
4. **Custom Templates**: Create templates for repetitive questions you ask frequently
5. **Export Early**: Download important conversations before clearing history
6. **Model Selection**: Match model to task (smaller for speed, larger for complex reasoning)
7. **Parameter Tuning**: Start with defaults, adjust only if needed
8. **Large Files**: PPTX and other large documents (up to 200 MB) are now supported — ideal for full slide decks and lengthy reports

## 🔐 Privacy & Data

- **All processing is LOCAL**: No data sent to external servers
- **Templates stored locally**: JSON files in `config/` directory
- **Main app writes no documents to disk**; the Library App stores extracted text and conversations in `libraries/` (git-ignored)
- **No telemetry**: Application doesn't track usage or collect analytics
- **Export control**: You control all exported HTML files

## 🤝 Contributing

To add new features or improve templates:

1. **Template Contributions**: Share your useful templates via Pull Request to `config/default_templates.json`
2. **Code Improvements**: Submit PRs with clear documentation
3. **Bug Reports**: Open issues with reproduction steps

## 📝 License

Copyright © 2026 Institute for Future Intelligence, Inc.

- **Code** (the apps, `rag_core/`, `library_ui/`, `utils/`, `tools/`, `tests/`, `config/`) is released under the [MIT License](LICENSE).
- **Training content** (the `docs/` site and the Markdown guides) is released under [Creative Commons Attribution 4.0 International](LICENSE-CC-BY-4.0) (CC BY 4.0).

## Funding

This material is based upon work supported by the National Science Foundation under Award Nos. 2600119 and 2600120. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

AI-STEPS is a collaboration between the Institute for Future Intelligence, the University of Hawaiʻi, Windward Community College and Kapiʻolani Community College.

## 🙏 Acknowledgments

- **Ollama**: For making local LLM deployment accessible
- **Streamlit**: For the excellent web framework
- **LangChain**: For LLM orchestration tools
- **Bootstrap**: For HTML export styling

## 📞 Support

For issues or questions:
1. Check this README first
2. Review the troubleshooting section
3. Check Ollama documentation: [ollama.ai/docs](https://ollama.ai/docs)
4. Check Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)

---

**Version**: 2.1
**Last Updated**: March 2026
**Author**: Scientific Computing Team
