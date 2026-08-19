# 🎯 RAG-First Application - User Guide (v4.0)

## Complete Architecture Change

This application is a **RAG-FIRST** tool. This means:
- 📄 **Primary purpose**: Analyze YOUR uploaded documents
- 🔒 **Default behavior**: Answers come ONLY from your documents
- ✅ **Opt-in for LLM**: Checkbox to enable general knowledge when needed

> **What changed in v4.0.** The three modes below work as they always have, but
> what happens underneath is different. Earlier versions pasted entire documents
> into the prompt, which meant anything past the context limit was silently
> discarded -- on a long document, most of it. There is now a real retrieval
> layer: documents are chunked, indexed, and *searched* per question, and only
> the relevant passages are sent. Answers carry `[S1]`-style citations traced to
> real pages, and the app verifies those citations rather than trusting them.
>
> The practical consequence: **document length no longer limits answer quality**,
> and "documents only" is now a promise the app can actually keep.

---

## Prerequisite: an embedding model

Document search needs an embedding model alongside your chat model:

```bash
ollama pull nomic-embed-text
```

Without one, documents can be uploaded and read but not searched. The app says
so explicitly rather than falling back to general knowledge while claiming to
read your files.

---

## Three Operating Modes

### Mode 1: STRICT RAG (Default when documents loaded) ✅

**When**: Documents uploaded + "Allow LLM General Knowledge" UNCHECKED

**Behavior**:
- ✅ AI answers ONLY from passages retrieved from your documents
- ✅ If the answer is not in them: "This information is not found in the provided documents"
- ✅ Cites each claim as `[S1]`, `[S2]`, traceable to a page or slide
- ✅ Fabricated citations are detected and flagged above the answer
- ✅ With nothing indexed, the app declines outright rather than answering from
  general knowledge -- it does not even call the model

**Visual Indicator**:
```
✅ STRICT RAG MODE: AI will answer ONLY from your uploaded documents
🎯 Perspective: Scientist | Audience: Scientist | Documents: 2 | Indexed chunks: 118 | Mode: RAG Only
```

**Sidebar Status**:
```
📄 Documents Indexed - RAG Mode Active
Answer source: Uploaded documents
☐ Allow LLM General Knowledge (UNCHECKED)
✅ AI will answer ONLY from documents

🧠 Vector store: 118 chunks from 2 document(s)
   [🗑️ Clear Vector Store]
```

**Checking the answer**: expand **"📚 Sources"** to read the exact passages
behind each citation, and **"🔬 Retrieval Diagnostics"** to see what was
searched and how much of the context budget was used.

**Use for**:
- Analyzing research papers
- Extracting specific information
- Verifying document content
- Literature review
- Compliance review

### Mode 2: HYBRID (Optional) ⚠️

**When**: Documents uploaded + "Allow LLM General Knowledge" CHECKED

**Behavior**:
- 📄 AI prioritizes document content
- 🧠 Can supplement with general knowledge
- ⚠️ Clearly distinguishes sources
- Useful when you want context beyond the document

**Visual Indicator**:
```
⚠️ HYBRID MODE: AI will use documents + general knowledge
🎯 Perspective: Scientist | Audience: Scientist | Documents: 2 | Mode: Hybrid
```

**Sidebar Status**:
```
📄 Documents Loaded - RAG Mode Active
Answer source: Uploaded documents
☑️ Allow LLM General Knowledge (CHECKED)
⚠️ AI can use general knowledge + documents
```

**Use for**:
- Comparing document to field standards
- Adding context to document findings
- Explaining concepts mentioned in document

### Mode 3: LLM MODE (No documents) 💬

**When**: No documents uploaded

**Behavior**:
- 🧠 AI uses its general knowledge
- Functions as a standard LLM interface
- No document constraints
- Can answer any question from training data

**Visual Indicator**:
```
💬 LLM MODE: AI will use general knowledge (upload documents for RAG mode)
🎯 Perspective: Scientist | Audience: Scientist
```

**Sidebar Status**:
```
📝 No Documents - LLM Mode
Answer source: LLM general knowledge
*Upload documents to enable RAG mode*
```

**Use for**:
- General questions
- Concept explanations
- Brainstorming
- When you don't have documents

---

## Quick Start

### Typical Workflow (RAG Mode)

1. **Upload Document**
   - Click "📄 Upload Documents (PDF, DOCX)"
   - Choose your file(s)
   - Click "📥 Process Documents"

2. **Verify Mode**
   - Look for green banner: "✅ STRICT RAG MODE"
   - Sidebar shows: "✅ AI will answer ONLY from documents"

3. **Ask Questions**
   - Type: "What are the main findings?"
   - Type: "Summarize the methodology"
   - Type: "What limitations are mentioned?"

4. **Get Document-Only Answers**
   - Answers cite specific sections
   - Direct quotes from document
   - "Not found" when information missing

### Enabling Hybrid Mode (Optional)

1. **Load Documents** (as above)

2. **Check the Box**
   - Sidebar → "☑️ Allow LLM General Knowledge"

3. **Ask Enhanced Questions**
   - "What does the document say and how does this compare to standard practice?"
   - "Explain the concept mentioned in the document"

4. **Get Hybrid Answers**
   - Document content clearly marked
   - General knowledge clearly marked
   - Both sources distinguished

### Using LLM Mode (No Documents)

1. **Don't Upload Documents** (or clear them)

2. **Ask Any Question**
   - "Explain photosynthesis"
   - "What is machine learning?"
   - Works like standard ChatGPT/Claude

---

## The Checkbox Explained

### "Allow LLM General Knowledge" Checkbox

**Location**: Sidebar, under "Answer Source"

**Default State**: UNCHECKED (disabled)

**Purpose**: Controls whether AI can use general knowledge

**States**:

| Documents | Checkbox | Result |
|-----------|----------|--------|
| Loaded | ☐ Unchecked | STRICT RAG - document only |
| Loaded | ☑ Checked | HYBRID - documents + knowledge |
| Not loaded | N/A | LLM MODE - knowledge only |

**Important**: When no documents are uploaded, the checkbox is not shown (LLM mode is automatic).

---

## Benefits of RAG-First Design

### ✅ Eliminates Confusion
**Before**: "Is it using my document or its knowledge?"  
**After**: Clear mode indicators show exactly what's happening

### ✅ Guarantees Document Fidelity
**Before**: AI mixed sources unpredictably  
**After**: In RAG mode, ONLY document content used

### ✅ User Control
**Before**: Couldn't control when AI used general knowledge  
**After**: Explicit checkbox gives you full control

### ✅ No Hallucinations in RAG Mode
**Before**: AI might make up details  
**After**: In RAG mode, everything must be from document

### ✅ Clear Error Messages
**Before**: Vague "I don't know" responses  
**After**: "This information is not found in the provided document"

---

## Example Interactions

### Example 1: Strict RAG Mode

**Setup**:
- Uploaded research paper
- Checkbox UNCHECKED

**Question**: "What was the sample size?"

**Response**:
```
According to the Methods section on page 4, the study included 
150 participants. The document states: "A total of 150 adult 
volunteers (ages 18-65) were recruited through online advertisements 
between January and March 2023."
```

**What you get**: Direct answer from document with citation and quote.

### Example 2: Strict RAG - Missing Info

**Setup**:
- Uploaded research paper
- Checkbox UNCHECKED

**Question**: "What statistical software did they use?"

**Response**:
```
This information is not found in the provided document. The Methods 
section describes the statistical analyses performed but does not 
specify which software package was used.
```

**What you get**: Honest acknowledgment that info is missing.

### Example 3: Hybrid Mode

**Setup**:
- Uploaded research paper
- Checkbox CHECKED ✅

**Question**: "What is the methodology and how does it compare to standard approaches?"

**Response**:
```
FROM THE DOCUMENT:
The document describes using a randomized controlled trial with 
pre-test/post-test design. Specifically: "Participants were randomly 
assigned to treatment or control groups, with measurements taken at 
baseline and after 8 weeks."

FROM GENERAL KNOWLEDGE:
This is a strong methodological approach. RCTs are considered the gold 
standard for causal inference because randomization helps control for 
confounding variables. The pre-test/post-test design allows within-subject 
comparisons, increasing statistical power.
```

**What you get**: Both document content AND broader context, clearly labeled.

### Example 4: LLM Mode

**Setup**:
- No documents uploaded

**Question**: "What is a t-test?"

**Response**:
```
A t-test is a statistical test used to compare the means of two groups 
to determine if they are significantly different from each other. It's 
commonly used when you have a small sample size (typically n < 30) and 
assumes the data follows a normal distribution...
```

**What you get**: Standard LLM explanation using general knowledge.

---

## Template System

### Empty by Default

**Philosophy**: You know what questions you need to ask!

**Current state**: No pre-built templates

**Your role**: Create templates that match YOUR workflow

### Creating Templates

1. **Navigate** to "Manage Templates" page
2. **Click** "➕ Add Template" tab
3. **Fill in**:
   - **Name**: "Analyze Methodology"
   - **Category**: "Research" (or create your own)
   - **Template Text**: "Analyze the research methodology described in this document"
   - **Requires Input**: Check if you want user to provide content
   - **Description**: "Analyzes research methods section"
4. **Click** "Add Template"

### Template Best Practices

**For RAG-first usage**, templates should be document-focused:

✅ **Good RAG Templates**:
- "Summarize the key findings from this document"
- "Extract all statistical results mentioned"
- "List limitations acknowledged in the document"
- "Compare methodologies across these documents"

❌ **Not ideal for RAG**:
- "Explain the concept of X" (no document needed)
- "What are best practices for Y?" (general question)

### Suggested Template Categories

Based on RAG-first philosophy:

1. **Document Analysis**
   - Summarize document
   - Extract key points
   - Identify limitations

2. **Research Review**
   - Analyze methodology
   - Evaluate findings
   - Compare studies

3. **Data Extraction**
   - Extract statistics
   - List participants details
   - Find specific information

4. **Synthesis**
   - Compare documents
   - Identify themes
   - Find contradictions

---

## Troubleshooting

### "No documents uploaded!" error

**Problem**: Trying to ask question in RAG mode without documents

**Solutions**:
1. Upload documents first, OR
2. Check "Allow LLM General Knowledge" to use LLM mode

### AI still seems to use general knowledge (RAG mode)

**Verify**:
1. Green banner shows "✅ STRICT RAG MODE"?
2. Sidebar shows "✅ AI will answer ONLY from documents"?
3. Checkbox is UNCHECKED?

**If all yes and still happening**:
- Lower temperature to 0.1-0.2 in sidebar
- Try with a more explicit question
- Check what text was extracted (preview mode)

### Want to use both document and knowledge

**Solution**: Check the "Allow LLM General Knowledge" box

**Result**: Hybrid mode activated

### Template doesn't work as expected

**Remember**: Templates work in whatever mode you're in
- RAG mode → template gets document-only answer
- Hybrid mode → template gets hybrid answer
- LLM mode → template gets general answer

---

## Advanced Usage

### Document Management

**Keep Documents Loaded**: Documents stay loaded across questions
- Ask multiple questions about same document
- No need to re-upload

**Clear Documents**: Click "🗑️ Clear All Documents"
- Switches to LLM mode
- Useful when done with document analysis

**Multiple Documents**: Upload several at once
- Ask comparison questions
- Analyze multiple sources together

### Mode Switching

**RAG → Hybrid**: Check the box mid-conversation
- Previous answers stay the same
- New answers use hybrid mode

**Hybrid → RAG**: Uncheck the box
- Returns to strict document-only mode
- Great for verification

**RAG/Hybrid → LLM**: Clear documents
- Switches to general knowledge mode
- Use for non-document questions

---

## Best Practices

### For Research Analysis (Strict RAG)

1. Upload paper
2. Keep checkbox UNCHECKED
3. Ask specific questions about content
4. Verify answers against source
5. Export results

### For Learning (Hybrid Mode)

1. Upload difficult paper
2. CHECK the box for hybrid mode
3. Ask: "Explain concepts + provide context"
4. Get document summary + broader understanding
5. Uncheck box to verify what's actually in paper

### For General Work (LLM Mode)

1. Don't upload documents
2. Ask any questions
3. Use for brainstorming, explanations, etc.
4. Upload documents when you need analysis

---

## Summary

### This is a RAG-First App

✅ **Primary function**: Document analysis  
✅ **Default behavior**: Document-only answers  
✅ **User control**: Checkbox to enable general knowledge  
✅ **Clear modes**: Visual indicators show current state  
✅ **Flexible**: Can use as regular LLM when needed  

### Three Simple Rules

1. **Want document analysis?**  
   → Upload documents, keep checkbox unchecked

2. **Want document + context?**  
   → Upload documents, check the box

3. **Want general Q&A?**  
   → Don't upload documents (or clear them)

---

**That's it! A clean, clear, RAG-first application that does what you expect.** 🎯
