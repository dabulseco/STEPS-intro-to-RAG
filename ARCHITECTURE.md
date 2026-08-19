# Application Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LOCAL LLM SCIENTIFIC ASSISTANT                    │
│                              (Streamlit App)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
          ┌─────────▼─────────┐          ┌──────────▼──────────┐
          │   CHAT PAGE       │          │  TEMPLATE MGMT PAGE │
          │                   │          │                     │
          │ - Chat Input      │          │ - View Templates    │
          │ - Template Browse │          │ - Add Template      │
          │ - History Display │          │ - Delete Template   │
          │ - Export Buttons  │          │ - Reset to Defaults │
          └─────────┬─────────┘          └──────────┬──────────┘
                    │                               │
                    └────────────┬──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   SIDEBAR CONTROLS      │
                    │                         │
                    │ - Navigation            │
                    │ - Perspective Select    │
                    │ - Audience Select       │
                    │ - Model Settings        │
                    │ - Parameters Sliders    │
                    │ - Question History      │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼────────┐    ┌─────────▼──────────┐   ┌────────▼─────────┐
│ PROMPT BUILDER │    │ TEMPLATE MANAGER   │   │   HTML EXPORT    │
│                │    │                    │   │                  │
│ - Build System │    │ - Load Templates   │   │ - Markdown→HTML  │
│   Prompts      │    │ - Save Templates   │   │ - Bootstrap Style│
│ - Perspective  │    │ - Add Template     │   │ - Export Convo   │
│   Definitions  │    │ - Delete Template  │   │ - Export Q&A     │
│ - Audience     │    │ - Get by Category  │   │                  │
│   Definitions  │    │                    │   │                  │
└────────────────┘    └─────────┬──────────┘   └──────────────────┘
                                │
                      ┌─────────▼──────────┐
                      │   JSON STORAGE     │
                      │                    │
                      │ default_templates  │
                      │ question_templates │
                      └────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL DEPENDENCIES                           │
└─────────────────────────────────────────────────────────────────────────┘
        │                        │                         │
┌───────▼────────┐    ┌─────────▼──────────┐    ┌────────▼─────────┐
│    OLLAMA      │    │    LANGCHAIN       │    │   STREAMLIT      │
│                │    │                    │    │                  │
│ - Local LLM    │    │ - ChatOllama       │    │ - UI Framework   │
│ - Embeddings   │    │ - Message types    │    │ - Session State  │
│ - API Endpoint │    │ - Output Parser    │    │ - Components     │
└────────────────┘    └────────────────────┘    └──────────────────┘
        │
┌───────▼──────────┐  ┌────────────────────┐
│    CHROMADB      │  │     RANK-BM25      │
│ (in-memory only) │  │  (keyword index)   │
└──────────────────┘  └────────────────────┘
```

## Retrieval Engine (`rag_core/`)

Streamlit owns the UI; `rag_core` owns retrieval and imports no Streamlit at
all. The boundary is deliberate — the engine is portable to another application
unchanged.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    app.py  (Streamlit — all UI)                          │
│   widgets · session state · progress bars · sources & diagnostics panels │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  calls rag_core.pipeline.answer()
                                │  passes a plain `generate(...)` callback
┌───────────────────────────────▼─────────────────────────────────────────┐
│                   rag_core/  (no Streamlit, no persistence assumption)   │
│                                                                          │
│   INGEST                              QUERY                              │
│   ──────                              ─────                              │
│   document_processor.py               query_transformer.py               │
│     → structured blocks                 condense / rewrite / multi-query │
│           │                             HyDE / filter inference          │
│           ▼                                       │                      │
│   chunker.py                                      ▼                      │
│     child chunks (~400 tok)           store.py                           │
│     + parent windows                    ┌──────────────┬──────────────┐  │
│     + page/section metadata             │ Chroma       │ BM25         │  │
│           │                             │ (semantic)   │ (keyword)    │  │
│           ▼                             └──────┬───────┴──────┬───────┘  │
│   enrichment.py  (opt-in)                      └───── RRF ────┘          │
│     contextual headers                            │                      │
│           │                                       ▼                      │
│           ▼                             reranker.py  (optional)          │
│   embeddings.py                           LLM or cross-encoder           │
│     batched Ollama /api/embed                     │                      │
│           │                                       ▼                      │
│           └────────► store.add_chunks() context_assembler.py             │
│                                           token budget · [S1] markers    │
│                                                   │                      │
│                                                   ▼                      │
│                                         citations.py                     │
│                                           verify every [S#] marker       │
│                                                                          │
│   pipeline.py orchestrates all of the above — the single answer path      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Making it persistent** is one line in `rag_core/store.py`:
`chromadb.EphemeralClient()` → `chromadb.PersistentClient(path=...)`. Nothing
above that line knows or cares how long the store lives.

## Data Flow Diagram

```
USER INPUT
    │
    ├─ [Direct Chat Input] ──────────────────────┐
    │                                             │
    └─ [Template Selection] ─┐                   │
                              │                   │
        [Select Template]     │                   │
              │               │                   │
              ▼               │                   │
        ┌──────────────┐      │                   │
        │ Form Mode?   │      │                   │
        └──────┬───────┘      │                   │
               │              │                   │
        ┌──────┴──────┐       │                   │
        │             │       │                   │
    [Yes]         [No]        │                   │
        │             │       │                   │
        ▼             ▼       │                   │
    [Fill Form]   [Populate]  │                   │
        │         Chat Input   │                   │
        │             │        │                   │
        └─────────────┴────────┘                   │
                  │                                │
                  ▼                                │
            [User Query] ◄───────────────────────┘
                  │
                  ▼
        ┌──────────────────────────┐
        │  run_query()  (app.py)   │
        │  single path for BOTH    │
        │  chat and template input │
        └────────┬─────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ resolve_mode()           │
        │ index empty? → LLM mode  │
        │ strict / hybrid / llm    │
        └────────┬─────────────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
  [no index]           [indexed]
       │                    │
       │                    ▼
       │          ┌──────────────────────┐
       │          │ Transform query      │
       │          │ (condense/rewrite/…) │
       │          └────────┬─────────────┘
       │                   ▼
       │          ┌──────────────────────┐
       │          │ Hybrid search + RRF  │
       │          │ semantic + keyword   │
       │          └────────┬─────────────┘
       │                   ▼
       │          ┌──────────────────────┐
       │          │ Re-rank (optional)   │
       │          └────────┬─────────────┘
       │                   ▼
       │          ┌──────────────────────┐
       │          │ Assemble context     │
       │          │ to a TOKEN BUDGET    │
       │          │ → [S1] [S2] …        │
       │          └────────┬─────────────┘
       │                   │
       └─────────┬─────────┘
                 ▼
        ┌──────────────────────────┐
        │  Build system prompt     │
        │  - Perspective           │
        │  - Audience              │
        │  - GROUNDING RULES       │
        │    (strict/hybrid/llm)   │
        └────────┬─────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Send to LLM     │
        │  (via Ollama)    │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │  Receive Response        │
        │  + verify [S#] citations │
        └────────┬─────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Store in History │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Display + Export │
        └──────────────────┘
```

## Template System Flow

```
┌─────────────────────────────────────────────────────┐
│               TEMPLATE LIFECYCLE                     │
└─────────────────────────────────────────────────────┘

[Application Start]
        │
        ▼
┌──────────────────┐
│ Load Default     │
│ Templates?       │
└────────┬─────────┘
         │
    [First Run]
         │
         ▼
┌──────────────────┐      ┌──────────────────────┐
│ Copy Defaults to │─────▶│ question_templates   │
│ User Templates   │      │ .json created        │
└──────────────────┘      └──────────────────────┘
         │
         │
         ▼
┌──────────────────────────────────────────────┐
│        TEMPLATE OPERATIONS                   │
├──────────────────────────────────────────────┤
│                                              │
│  [View] ──▶ Load → Filter → Display         │
│                                              │
│  [Add]  ──▶ Form → Validate → Save          │
│                                              │
│  [Delete]─▶ Confirm → Remove → Save         │
│                                              │
│  [Use]  ──▶ Select → Mode? ──┬──▶ Form      │
│                               └──▶ Chat      │
│                                              │
│  [Reset]──▶ Confirm → Restore Defaults      │
│                                              │
└──────────────────────────────────────────────┘
```

## Key Components Interaction

```
┌──────────────┐
│   USER       │
└──────┬───────┘
       │
       │ Interacts with
       ▼
┌──────────────┐       Uses        ┌──────────────┐
│  STREAMLIT   │◄───────────────────│    UTILS     │
│     UI       │                    │              │
└──────┬───────┘                    │ - Templates  │
       │                            │ - Prompts    │
       │ Calls                      │ - HTML       │
       ▼                            └──────────────┘
┌──────────────┐
│  LANGCHAIN   │
│   WRAPPER    │
└──────┬───────┘
       │
       │ Sends Prompt
       ▼
┌──────────────┐
│   OLLAMA     │
│   API        │
└──────┬───────┘
       │
       │ Returns Response
       ▼
┌──────────────┐
│   DISPLAY    │
│   & EXPORT   │
└──────────────┘
```
