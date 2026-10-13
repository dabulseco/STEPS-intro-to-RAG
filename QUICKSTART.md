# 🚀 Quick Start Guide

## Installation (5 minutes)

### 1. Install Ollama
Visit [ollama.ai](https://ollama.ai) and download the installer for your operating system.

### 2. Install Python Dependencies
```bash
git clone https://github.com/dabulseco/STEPS-intro-to-RAG.git
cd STEPS-intro-to-RAG
pip install -r requirements.txt
```

### 3. Start Ollama and Download Models
```bash
# In terminal 1: Start Ollama
ollama serve

# In terminal 2: Pull a chat model (first time only)
ollama pull llama2

# And an embedding model -- required for document search
ollama pull nomic-embed-text
```

**Why two models?** They do different jobs. The chat model writes answers; the
embedding model turns your documents into searchable vectors. Without an
embedding model you can still chat and still upload documents, but the app
cannot search them -- and it will say so rather than quietly answering from
general knowledge.

### 4. Launch the App
```bash
# In terminal 2 (or a new terminal)
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`

## First Steps (2 minutes)

### 1. Configure Context (Left Sidebar)
- **Perspective**: Choose "Scientist" (who's writing)
- **Audience**: Choose "College Student" (who's reading)
- **Model**: Select "llama2" (or whichever model you downloaded)

### 2. Save Your First Template
The app starts with an **empty** template library -- you build it from the
questions you actually ask.

1. Go to **"Manage Templates"** (left sidebar) → **"➕ Add Template"**
2. Name: `Explain Scientific Concept`
3. Category: `Teaching`
4. Template text: `Explain this scientific concept clearly:`
5. Check **"Requires user input"**, then click **"➕ Add Template"**
6. Back on the Chat page, open **"📋 Question Templates"**, pick it, click
   **"📝 Use Template with Form"**, enter `photosynthesis`, and submit.

### 3. Try Free-Form Chat
Just type in the chat input: "Explain how CRISPR works"

### 4. Search Your Own Documents
1. Open **"📄 Upload Documents"** on the main page
2. Choose a PDF, DOCX or PPTX and click **"📥 Process Documents"**
3. Watch the progress bar as chunks are embedded
4. The sidebar now shows **🧠 Vector store: N chunks**
5. Ask a question about the document

The app searches your documents and answers only from what it finds, citing
each passage as `[S1]`, `[S2]`. Expand **"📚 Sources"** under any answer to see
the exact text, page number and section behind each citation.

**Your documents are never written to disk.** The index lives in memory and
disappears when you close the app, or when you press **"🗑️ Clear Vector
Store"** in the sidebar.

## Key Features to Explore

### ✅ Change Who's Talking
Try different combinations:
- **Scientist → High School Student**: Technical but accessible
- **Teacher → Middle School Student**: Very clear, age-appropriate
- **Lay Person → Scientist**: Informal but detailed

### ✅ Browse All Templates
Navigate to **"Manage Templates"** (left sidebar) to manage your library. The
category list is ready to organize into, even though it starts empty:
- Research (papers, methods, critique)
- Teaching (lessons, analogies, experiments)
- Laboratory (protocols, safety, troubleshooting)
- Statistics (tests, interpretation)
- And more!

⚠️ **Avoid "Reset to Defaults"** unless you have backed up
`config/question_templates.json`. The defaults file is empty, so reset deletes
your templates and restores nothing.

### ✅ Create Your Own Template
1. Go to **"Manage Templates"** → **"➕ Add Template"**
2. Fill in:
   - Name: "Explain Like I'm Five"
   - Category: "Custom"
   - Template: "Explain this concept in the simplest possible terms:"
   - Check "Requires user input"
3. Click **"➕ Add Template"**
4. Use it immediately from the Chat page!

### ✅ Export Your Work
- After any conversation, click **"⬇️ Download Full Conversation"**
- Get professional HTML with Bootstrap styling
- Perfect for sharing or keeping records

## Tips for Best Results

### 🎯 For Teaching
- **Perspective**: "High School Teacher"
- **Audience**: Match your students' level
- **Templates**: Use "Create Analogy" and "Design Experiment"

### 🎯 For Research
- **Perspective**: "Scientist"
- **Audience**: "Scientist" or "College Student"
- **Templates**: Use "Summarize Research Paper" and "Critique Study"

### 🎯 For Lab Work
- **Perspective**: "Scientist"
- **Audience**: Match your team's expertise
- **Templates**: Use "Design Laboratory Protocol" and "Troubleshoot Experiment"

## Common Questions

**Q: Can I change perspective mid-conversation?**  
A: The current version applies perspective/audience to the entire conversation. To switch, start a new conversation (click "Clear History").

**Q: Where are my templates saved?**  
A: In `config/question_templates.json` — back this file up, it is the only copy.

**Q: Why does the app start with no templates?**  
A: `config/default_templates.json` ships empty by design. Templates you add go
to `question_templates.json`. If you want a set restored by "Reset to Defaults",
copy your finished templates into `default_templates.json`.

**Q: Can I use this offline?**  
A: Yes! Everything runs locally. No internet needed after installation.

**Q: Which model should I use?**  
A: Start with `llama2` (good balance of speed and quality). Try `mistral` for coding questions.

**Q: What do Temperature, Top-K and Top-P do?**  
A: They control how the model picks each next word. For document questions set
**Temperature to 0.1–0.3** and leave Top-K (40) and Top-P (0.9) alone — they
overlap with temperature, and changing several at once makes results impossible
to attribute. Full explanations and three recommended presets are in
[`docs/settings.html`](docs/settings.html).

**Q: How do I empty the vector store?**  
A: **"🗑️ Clear Vector Store"** in the sidebar drops the index and the loaded
documents. It also clears itself when you close the app — nothing is written to
disk at any point.

**Q: Why are responses slow?**  
A: Use a smaller model variant (e.g., `llama2:7b`), and turn off options under
**"🧪 Advanced Techniques"** -- each one adds a model call. Turning off
**"Generate follow-up questions"** saves a call on every answer.

Note: do *not* lower "Context Length" to speed things up when documents are
loaded. That shrinks the budget for retrieved evidence, so fewer passages reach
the model and answers get worse.

**Q: Do I have to re-upload my documents every time?**  
A: Yes. The vector store is deliberately temporary -- it lives in memory only
and is gone when the app closes. Nothing is written to disk.

**Q: The app says my documents aren't indexed. What now?**  
A: You are missing an embedding model. Run `ollama pull nomic-embed-text` and
process the documents again.

**Q: What do the [S1] markers in answers mean?**  
A: They are citations. Each points to a retrieved passage listed under the
**"📚 Sources"** panel below the answer. If the app ever cites a number that
was not supplied, it flags the answer with a warning.

## Tuning Retrieval (optional)

> 📖 **Full reference:** every control below is documented in detail — with
> ranges, defaults and the effect of changing each one — in the documentation
> website at [`docs/`](docs/index.html). Open `docs/index.html` in a browser.

Once documents are loaded, two sidebar panels appear.

**🎛️ Retrieval Tuning** — these change *which* passages are found and cost no
extra model call, so they are free to experiment with:

| Control | Range · default | What it does |
|---|---|---|
| Semantic ↔ Keyword balance | 0.0–1.0 · **0.5** | Weights meaning-based against exact-wording search. Toward keyword for part numbers and defined terms; toward semantic for paraphrased questions. |
| Passages sent to the model | 1–15 · **5** | How much evidence reaches the model. Raise for broad synthesis questions, lower for narrow lookups. |
| Candidates considered | up to 40 · **12** | The pool the final selection draws from. Raise to 20–25 when re-ranking is on. |
| Candidates per index | 5–50 · **20** | How deep each index searches before results merge. Raise if passages are missed entirely. |
| Restrict search to | all documents | Limits retrieval to chosen files, applied *before* searching. |
| Generate follow-up questions | **on** | Suggests 3 follow-ups. Costs one model call per answer — turn off for speed. |

**🧪 Advanced Techniques** -- each costs an extra model call, so turn on only
what helps your documents:

| Option | Turn it on when |
|---|---|
| Resolve follow-up references | *(on by default)* you ask follow-ups like "how does that compare?" |
| Rewrite query for retrieval | your wording differs from the documents' terminology |
| Search multiple phrasings | you are missing passages you know exist |
| HyDE | questions are short and jargon-heavy |
| Infer document/section filter | you name a document or section in the question |
| Re-rank retrieved passages | the right passage is found but ranked too low |
| Add contextual headers *(at upload)* | a document leans on pronouns and back-references — costs one model call **per chunk**, so minutes on a long document |

Not sure which help? Run the evaluation harness against your own documents:

```bash
python tools/eval_retrieval.py --docs path/to/docs --questions golden.json
```

## Next Steps

1. ✅ Explore all template categories
2. ✅ Create 2-3 custom templates for your common questions
3. ✅ Try different perspective/audience combinations
4. ✅ Upload a document and inspect the 📚 Sources behind an answer
5. ✅ Experiment with model parameters
5. ✅ Export and save your important conversations

## Need Help?

- 📖 Full documentation: See `README.md`
- 🐛 Troubleshooting: Check the Troubleshooting section in README
- 💡 Model parameters: See "Model Parameters Explained" in README

**Happy exploring! 🔬**
