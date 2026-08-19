# 🚀 Quick Start Guide

## Installation (5 minutes)

### 1. Install Ollama
Visit [ollama.ai](https://ollama.ai) and download the installer for your operating system.

### 2. Install Python Dependencies
```bash
cd ollama_app
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

### 2. Try a Template
1. Click **"📋 Question Templates"** (main page)
2. Select category: **"Teaching"**
3. Choose: **"Explain Scientific Concept"**
4. Click **"📝 Use Template with Form"**
5. Enter a concept like: "photosynthesis"
6. Click **"🚀 Submit Question"**

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
Navigate to **"Manage Templates"** (left sidebar) to see all 30+ templates organized by:
- Research (papers, methods, critique)
- Teaching (lessons, analogies, experiments)
- Laboratory (protocols, safety, troubleshooting)
- Statistics (tests, interpretation)
- And more!

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
A: In `config/question_templates.json` - you can backup this file!

**Q: Can I use this offline?**  
A: Yes! Everything runs locally. No internet needed after installation.

**Q: Which model should I use?**  
A: Start with `llama2` (good balance of speed and quality). Try `mistral` for coding questions.

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

Once documents are loaded, two sidebar panels appear:

**🎛️ Retrieval Tuning** -- the one setting worth knowing is the
**semantic ↔ keyword balance**. Slide it toward keyword when exact terms matter
(part numbers, gene names, statute references); toward semantic for
conceptual or paraphrased questions. The default 50/50 is a good starting point.

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
