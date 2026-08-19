# Troubleshooting Guide

## Quick Diagnostics

### ✅ Is Everything Working?

Run through this checklist:

```bash
# 1. Check if Ollama is running
curl http://localhost:11434/api/tags

# 2. Check if you have models
ollama list

# 3. Check Python packages
pip list | grep -E "streamlit|langchain|requests|chromadb|rank-bm25"

# 4. Check you have an embedding model (needed for document search)
ollama list | grep -E "embed"
# If nothing: ollama pull nomic-embed-text

# 5. Try running the app
cd ollama_app
streamlit run app.py

# 6. Run the retrieval unit tests (no Ollama needed, ~1 second)
python -m pytest tests/ -q
```

If all work, you're good! If not, see specific issues below.

---

## Common Issues & Solutions

### 🔴 Issue 1: "No models available" Error

**Symptoms**:
- Sidebar shows "No models available"
- Red error message
- Can't select a model

**Causes & Solutions**:

#### Cause A: Ollama Not Running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If it fails, start Ollama
ollama serve
```

#### Cause B: No Models Downloaded
```bash
# Check what models you have
ollama list

# If empty, pull a model
ollama pull llama2
# or
ollama pull mistral
```

#### Cause C: Wrong Ollama Port
- Default is `localhost:11434`
- If you changed it, update `app.py`:
```python
# Line ~320-325 in app.py
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:YOUR_PORT",  # Change this
    ...
)
```

---

### 🔴 Issue 2: Templates Not Loading

**Symptoms**:
- Template dropdown is empty
- "No templates available" message
- Can't see default templates

**Solutions**:

#### Solution A: Delete and Regenerate
```bash
cd ollama_app
rm config/question_templates.json
# Restart app - it will auto-create from defaults
streamlit run app.py
```

#### Solution B: Check File Permissions
```bash
# Make sure config directory is writable
chmod -R 755 config/
```

#### Solution C: Manually Copy Defaults
```bash
cd ollama_app/config
cp default_templates.json question_templates.json
```

#### Solution D: Check JSON Syntax
If you edited the JSON file manually:
```bash
# Validate JSON
python3 -c "import json; json.load(open('config/question_templates.json'))"
```

---

### 🔴 Issue 3: Slow Response Times

**Symptoms**:
- Waiting 30+ seconds for responses
- App freezes during generation
- High CPU usage

**Solutions**:

#### Solution A: Use Smaller Model
```bash
# Instead of llama2:13b or llama2:70b, use:
ollama pull llama2:7b
# or
ollama pull mistral:7b
```

#### Solution B: Turn off optional model calls
Each option under **🧪 Advanced Techniques** costs an extra model call per
question, and follow-up generation costs one more. With everything on, a
question can take 3-4x as long. Turn off what you are not benefiting from --
**"🔬 Retrieval Diagnostics"** on each answer shows which stages ran.

> ⚠️ **Do not lower "Context Length" to speed up document questions.** That
> setting sizes the budget for retrieved evidence, so lowering it sends the
> model *less* of your document and makes answers worse. It is safe to lower
> only when no documents are loaded.

#### Solution C: Adjust Temperature
- Lower temperature = faster but less creative
- Try setting to 0.1 for speed

#### Solution D: Check System Resources
```bash
# Check memory usage
free -h

# Check CPU usage
top

# Close other heavy applications
```

#### Solution E: Limit Conversation History
In `app.py`, line ~118, change:
```python
# From:
for past_query, past_response in chat_history[-5:]:
# To:
for past_query, past_response in chat_history[-3:]:  # Only last 3
```

---

### 🔴 Issue 3b: Document Questions Answered From General Knowledge

**Symptoms**:
- Banner reads "⚠️ DOCUMENTS NOT INDEXED"
- Sidebar shows "Documents extracted but NOT indexed"
- Answers ignore your uploaded files

**Cause**: No embedding model. Extraction and *search* are separate steps, and
search needs a model your chat model cannot provide.

**Solution**:
```bash
ollama pull nomic-embed-text
```
Then press "📥 Process Documents" again. The sidebar should show
**🧠 Vector store: N chunks**.

---

### 🔴 Issue 3c: Retrieval Misses Something You Know Is There

**Symptoms**:
- "This information is not found in the provided documents", but you can see it
- The right passage never appears under 📚 Sources

**Diagnose in this order** -- each step rules out one layer:

1. **Was the text extracted?** Open **"🔍 Inspect Indexed Chunks"**. If the
   passage is not there, the problem is extraction, not search. Scanned PDFs and
   text inside images yield nothing.
2. **What was actually searched for?** Open **"🔬 Retrieval Diagnostics"** on
   the answer. If a query rewrite mangled your question, turn off
   "Rewrite query for retrieval".
3. **Was it found but ranked too low?** If diagnostics show many hits, raise
   "Passages sent to the model", or turn on "Re-rank retrieved passages".
4. **Wrong search method?** Exact identifiers (part numbers, gene names) need
   the balance slider toward **keyword**; paraphrased questions need
   **semantic**.
5. **Did it get cut for budget?** Diagnostics reports passages dropped for the
   context budget. Raise "Context Length".

**Measure rather than guess** -- run the evaluation harness on your own corpus:
```bash
python tools/eval_retrieval.py --docs path/to/docs --questions golden.json
```
where `golden.json` is `[{"question": "...", "expect": "text that must appear"}]`.
It reports hit rate and MRR per configuration, so you can see whether a setting
helps generally or only on the one question you happened to test.

---

### 🔴 Issue 3d: Answer Cites a Source That Does Not Exist

**Symptoms**: A warning above the answer, e.g. "⚠️ Cites [S7], which was never
provided."

**Cause**: The model invented a citation. The app detects this by checking every
`[S#]` marker against the sources actually supplied.

**Solutions**:
- Lower Temperature (0.1-0.3)
- Use a larger chat model -- small models fabricate citations more often
- Reduce "Passages sent to the model"; long source lists confuse numbering

---

### 🔴 Issue 3e: Documents Disappear After Restarting

**Not a bug.** The vector store is deliberately in-memory only: it is cleared
when you press "🗑️ Clear Vector Store" and when the app closes, and nothing is
written to disk. Re-upload and re-process to search them again.

If you need documents to persist between sessions, that is a change to one line
in `rag_core/store.py` (`EphemeralClient` → `PersistentClient(path=...)`).

---

### 🔴 Issue 4: Import Errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'langchain_ollama'
ModuleNotFoundError: No module named 'streamlit'
```

**Solution**:
```bash
cd ollama_app
pip install -r requirements.txt

# If still fails, install individually:
pip install streamlit
pip install langchain-ollama
pip install langchain-core
pip install requests
pip install markdown
pip install PyPDF2 python-docx python-pptx   # document extraction
pip install chromadb rank-bm25               # retrieval
```

**`ModuleNotFoundError: No module named 'chromadb'` or `'rank_bm25'`** means the
retrieval engine cannot start. **`No module named 'pptx'`** means PPTX uploads
will fail while PDF and DOCX still work.

**`ImportError` mentioning `sentence-transformers`** appears only if you selected
the cross-encoder re-ranker. Either install it (`pip install
sentence-transformers`) or switch the re-ranker back to "Local LLM", which needs
no extra package.

---

### 🔴 Issue 5: Template Form Not Submitting

**Symptoms**:
- Click "Submit Question" but nothing happens
- Form doesn't clear after submission
- No error message

**Solutions**:

#### Solution A: Check if Content Required
- If template requires input, make sure you filled the text area
- Look for red error message: "Please provide the required input"

#### Solution B: Check Model Selection
- Make sure a model is selected in sidebar
- If no model selected, you'll see error about no model

#### Solution C: Browser Cache
```bash
# Clear Streamlit cache
streamlit cache clear

# Or restart with cache clearing
streamlit run app.py --server.runOnSave false
```

---

### 🔴 Issue 6: HTML Export Issues

**Symptoms**:
- Download button doesn't work
- Downloaded file is empty
- Can't open exported HTML

**Solutions**:

#### Solution A: Browser Download Settings
- Check if browser blocked the download
- Look for blocked popup notification
- Allow downloads from localhost

#### Solution B: File Permissions
```bash
# Check if you can write to downloads folder
touch ~/Downloads/test.txt
rm ~/Downloads/test.txt
```

#### Solution C: Try Different Browser
- Chrome/Edge: Usually works best
- Firefox: Sometimes has issues with blob downloads
- Safari: May need to allow downloads

#### Solution D: Check HTML Content
If file downloads but won't open:
```bash
# Check if file has content
ls -lh ~/Downloads/conversation_transcript.html

# Open in text editor to verify it's valid HTML
```

---

### 🔴 Issue 7: Perspective/Audience Not Working

**Symptoms**:
- Responses don't match selected perspective/audience
- Same answers regardless of settings
- No difference between combinations

**Explanation**:
This might not be a bug! Some possibilities:

#### Possibility A: Model Limitation
- Smaller models may not fully grasp context differences
- Try a larger model (e.g., llama2:13b instead of llama2:7b)

#### Possibility B: Question Too Simple
- Very simple questions get similar answers regardless
- Try more complex questions to see differences

#### Possibility C: Need Clearer Instructions
Try adding explicit language in your questions:
```
Original: "Explain photosynthesis"
Better: "Explain photosynthesis in detail appropriate for my audience"
```

#### Solution: Verify It's Working
Compare these with Scientist→Scientist vs Teacher→Middle School Student:
- "Explain quantum entanglement"
- "How does CRISPR work?"
- "What causes climate change?"

You should see clear differences in:
- Vocabulary complexity
- Detail level
- Use of examples/analogies

---

### 🔴 Issue 8: Page Navigation Not Working

**Symptoms**:
- Can't switch between Chat and Manage Templates
- Radio buttons don't respond
- Stuck on one page

**Solution**:
```bash
# Clear Streamlit cache and restart
streamlit cache clear
streamlit run app.py
```

---

### 🔴 Issue 9: Conversation History Lost

**Symptoms**:
- History disappears after closing browser
- Can't find previous conversations
- History resets unexpectedly

**Explanation**:
This is expected behavior! Streamlit session state is temporary.

**Solutions**:

#### Solution A: Export Before Closing
- Always click "Download Full Conversation" before closing
- Save the HTML file for permanent records

#### Solution B: Don't Refresh Page
- Avoid refreshing browser (F5)
- Use the app's navigation instead

#### Solution C: Keep Browser Open
- Minimize window instead of closing
- Use browser tabs to keep session alive

---

### 🔴 Issue 10: Can't Delete Templates

**Symptoms**:
- Delete button doesn't work
- Template still appears after deletion
- Error when trying to delete

**Solutions**:

#### Solution A: Check File Permissions
```bash
cd ollama_app
chmod 644 config/question_templates.json
```

#### Solution B: Manual Deletion
Edit `config/question_templates.json`:
1. Open in text editor
2. Find and remove the template object
3. Save file
4. Restart app

#### Solution C: Reset to Defaults
- Go to Manage Templates → Settings
- Click "Reset to Default Templates"
- Start fresh

---

## 🔧 Advanced Troubleshooting

### Debug Mode

Add debug output to see what's happening:

```python
# In app.py, add at the top (after imports):
import logging
logging.basicConfig(level=logging.DEBUG)

# Then add debug prints where needed:
st.write("Debug: Selected template:", st.session_state.selected_template)
st.write("Debug: Perspective:", st.session_state.perspective)
```

### Check Logs

```bash
# Streamlit logs
streamlit run app.py --logger.level debug

# Ollama logs (if running as service)
journalctl -u ollama -f

# Or check Ollama output in terminal where you ran 'ollama serve'
```

### Test Components Individually

#### Test Template Manager:
```python
python3 -c "
from utils.template_manager import TemplateManager
tm = TemplateManager()
templates = tm.load_templates()
print(f'Loaded {len(templates)} templates')
print('Categories:', tm.get_all_categories())
"
```

#### Test Prompt Builder:
```python
python3 -c "
from utils.prompt_builder import PromptBuilder
prompt = PromptBuilder.build_system_prompt('Scientist', 'High School Student')
print(prompt)
"
```

#### Test HTML Export:
```python
python3 -c "
from utils.html_export import build_pair_html
html = build_pair_html(0, 'Test question', 'Test answer')
print('HTML length:', len(html))
print('Has Bootstrap:', 'bootstrap' in html.lower())
"
```

#### Test the Retrieval Engine:
```bash
# Full unit suite -- no Ollama required, runs in about a second
python -m pytest tests/ -q

# End-to-end retrieval against a synthetic corpus (needs Ollama + an
# embedding model). Compares semantic, keyword and hybrid search.
python tools/eval_retrieval.py --demo
```

#### Test Embedding Connectivity:
```python
python3 -c "
from rag_core.embeddings import OllamaEmbedder
from rag_core.ollama_api import list_embedding_models
print('Embedding models available:', list_embedding_models())
e = OllamaEmbedder(model='nomic-embed-text')
v = e.embed_query('hello world')
print('Vector dimension:', len(v))
"
```

#### Test Chunking of a Real File:
```python
python3 -c "
from utils.document_processor import DocumentProcessor as DP
blocks = DP.extract_blocks_from_pdf(open('yourfile.pdf','rb').read())
print('blocks:', len(blocks))
for b in blocks[:3]:
    print(' page', b['page'], '|', b['section'], '|', str(b['text'])[:60])
"
```
If this prints no blocks, the problem is extraction (likely a scanned PDF), not
retrieval.

---

## 📱 Platform-Specific Issues

### macOS

**Issue**: "zsh: command not found: ollama"
```bash
# Add to ~/.zshrc
export PATH="/usr/local/bin:$PATH"
source ~/.zshrc
```

**Issue**: Port already in use
```bash
# Check what's using port 11434
lsof -i :11434

# Kill the process if needed
kill -9 [PID]
```

### Windows

**Issue**: Ollama won't start
- Make sure you ran the installer
- Try running as Administrator
- Check Windows Defender isn't blocking it

**Issue**: Python not found
```bash
# Use py instead of python3
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

### Linux

**Issue**: Permission denied
```bash
# Add user to ollama group
sudo usermod -aG ollama $USER
# Log out and back in
```

**Issue**: Systemd service
```bash
# Start service
sudo systemctl start ollama

# Enable on boot
sudo systemctl enable ollama

# Check status
sudo systemctl status ollama
```

---

## 🆘 Still Having Issues?

### Information to Gather

Before seeking help, collect:

1. **System Info**:
```bash
python3 --version
streamlit --version
ollama --version  # or ollama -v
uname -a  # Linux/Mac
```

2. **Error Messages**:
- Full error text from terminal
- Browser console errors (F12 → Console)
- Screenshots if UI issue

3. **What You Tried**:
- Which solutions from this guide
- Any modifications to code
- Recent changes before issue started

4. **Environment**:
- Operating system and version
- Available RAM and CPU
- Models installed (`ollama list`)

### Where to Get Help

1. **Ollama Issues**: [github.com/ollama/ollama/issues](https://github.com/ollama/ollama/issues)
2. **Streamlit Issues**: [github.com/streamlit/streamlit/issues](https://github.com/streamlit/streamlit/issues)
3. **LangChain Issues**: [github.com/langchain-ai/langchain/issues](https://github.com/langchain-ai/langchain/issues)

---

## 🎓 Best Practices to Avoid Issues

1. **Regular Updates**:
```bash
# Update Ollama
ollama pull llama2  # Re-pull to get latest

# Update Python packages
pip install --upgrade streamlit langchain-ollama langchain-core
```

2. **Backup Templates**:
```bash
# Before resetting or making big changes
cp config/question_templates.json config/question_templates_backup.json
```

3. **Export Important Conversations**:
- Don't rely on browser history
- Always export before closing app

4. **Monitor Resources**:
- Watch CPU/RAM usage
- Close unused apps when running large models
- Consider upgrading RAM if consistently slow

5. **Test After Changes**:
- If you modify code, test immediately
- Make small changes, test each one
- Keep backup of working version

---

## 📋 Quick Reference Commands

```bash
# Start everything
ollama serve                           # Terminal 1
cd ollama_app && streamlit run app.py  # Terminal 2

# Check status
curl http://localhost:11434/api/tags   # Ollama running?
ollama list                            # Models available?
ls -la config/                         # Templates exist?

# Reset everything
rm config/question_templates.json      # Delete user templates
streamlit cache clear                  # Clear app cache
# Restart app

# Debug mode
streamlit run app.py --logger.level debug

# Test components
python3 -m utils.template_manager
python3 -m utils.prompt_builder
python3 -m utils.html_export
```

---

**Remember**: Most issues have simple solutions. Work through this guide systematically, and you'll likely find the fix! 🔧
