# 🎯 Improved RAG Mode - Document-Focused Responses

## What Changed

The application now has **strict RAG (Retrieval Augmented Generation) mode** that ensures the AI responds ONLY from your uploaded documents, not from its general knowledge.

---

## The Problem (Before)

When you uploaded a document and asked questions, the AI would:
- ❌ Mix document content with its general knowledge
- ❌ Add information not in the document
- ❌ Speculate or infer beyond what was stated
- ❌ Use training data to "enhance" answers

**Example**:
```
You: "What methodology did they use?"
AI: "They used X methodology. This is a common approach in the field because..."
     ↑ From document    ↑ From AI's general knowledge (NOT from document!)
```

---

## The Solution (Now)

When documents are uploaded, the AI enters **strict RAG mode**:
- ✅ Answers ONLY from the uploaded document
- ✅ Cites specific sections and pages
- ✅ States clearly when information is NOT in the document
- ✅ No speculation or inference beyond the text
- ✅ Uses direct quotes when possible
- ✅ Only adds external context if explicitly requested

**Example**:
```
You: "What methodology did they use?"
AI: "According to page 3 of the document, they used X methodology. 
     The document states: '[direct quote]'. No additional methodological 
     details are provided in this document."
     ↑ All from the document, clearly cited
```

---

## How It Works

### Visual Indicator

When you upload documents, you'll see:

```
📄 RAG MODE ACTIVE: AI will answer based on your uploaded document(s) only
🎯 Perspective: Scientist | Audience: Scientist | Documents: 1
```

This confirms the AI is in strict document-focused mode.

### System Prompt Changes

The AI receives these critical instructions when documents are present:

```
CRITICAL DOCUMENT-FOCUSED INSTRUCTIONS:

1. ANSWER ONLY FROM THE DOCUMENT
   - Base responses EXCLUSIVELY on uploaded document content
   - Do NOT use general knowledge unless explicitly asked

2. CITE SPECIFIC SECTIONS
   - Reference specific parts (e.g., "According to page 3...")
   - Use page numbers, sections, or direct quotes

3. ACKNOWLEDGE LIMITATIONS
   - If information is not in the document, state:
     "This information is not found in the uploaded document(s)."

4. NO SPECULATION
   - Do not infer or assume beyond what's explicitly stated
   - If ambiguous, acknowledge the ambiguity

5. EXTERNAL CONTEXT ONLY WHEN REQUESTED
   - Only provide general knowledge if explicitly asked
   - Examples: "compare this to other research" or "what's the broader context"

6. DIRECT QUOTES
   - Use direct quotes to support answers
   - Mark them clearly as quotes

7. DOCUMENT BOUNDARIES
   - Distinguish between what IS vs IS NOT in the document
```

---

## Updated Templates

All 8 document analysis templates have been updated to reinforce document-only responses:

### Before
```
"Summarize this paper and provide the significance of their findings."
```

### After
```
"Based ONLY on the uploaded document content, provide a comprehensive summary 
including:
1. Main topic and purpose
2. Key findings (cite specific sections)
3. Important conclusions stated in the document
4. Significance as described in the document

Do NOT add information from outside the document."
```

---

## Usage Examples

### ✅ Good RAG Queries

**Direct questions**:
```
"What does the document say about X?"
"Summarize the methodology section"
"What conclusions do the authors draw?"
"Are there any limitations mentioned?"
```

**With citations**:
```
"Quote the main finding from the results section"
"What specific data is provided in Table 1?"
"List the keywords mentioned in the abstract"
```

**Boundary acknowledgment**:
```
"Does this document discuss Y?"
→ AI: "No, topic Y is not discussed in this document."

"What sample size did they use?"
→ AI: "The document does not specify a sample size."
```

### ❌ What to Avoid

**Asking for external knowledge** (unless intentional):
```
❌ "What is generally known about X?"
   → Will get: "This information is not in the document"
   
✅ "What does THIS document say about X?"
   → Will get: Document-specific answer
```

**Expecting speculation**:
```
❌ "What might be the implications?"
   → Will get: Only implications stated in document
   
✅ "What implications do the AUTHORS discuss?"
   → Will get: Author's stated implications
```

---

## Asking for External Context

If you DO want the AI to use external knowledge, be explicit:

### Request External Information
```
"Based on the document, they used X method. How does this compare to 
 standard practice in the field?"
```

### Request Broader Context
```
"The document mentions Y finding. What is the broader scientific context 
 for this finding?"
```

### Request Comparison
```
"Compare the findings in this document to current literature on the topic"
```

The AI will then provide both document-based and external information, clearly distinguished.

---

## Testing RAG Mode

### Quick Test

1. **Upload a document** (PDF or DOCX)
2. **Ask a specific question** about document content
3. **Verify the response**:
   - Does it cite page numbers or sections?
   - Does it use quotes from the document?
   - Does it stay within document boundaries?

### Example Test

**Document**: Upload a research paper  
**Question**: "What was the sample size?"  

**Good Response** (Strict RAG):
```
"According to the Methods section on page 4, the study included 
127 participants. The document states: 'A total of 127 adult 
volunteers were recruited...'"
```

**Bad Response** (Not strict RAG):
```
"The study used 127 participants, which is a moderate sample size 
for this type of research. Typically, studies in this field use 
between 100-200 participants..."
↑ Added external context not in document
```

---

## Benefits

### For Researchers
✅ **Accurate citation**: Know exactly what's in the paper  
✅ **No hallucinations**: AI won't make up details  
✅ **Verification**: Can check every claim against source  
✅ **Trust**: Responses are faithful to the document  

### For Students
✅ **Reading comprehension**: Understand what authors actually said  
✅ **No confusion**: Clear distinction between document and general knowledge  
✅ **Study accuracy**: Learn from the actual source material  

### For Document Analysis
✅ **Precision**: Extract exactly what's stated  
✅ **Objectivity**: No AI bias or interpretation added  
✅ **Transparency**: Every claim is traceable to source  

---

## Technical Details

### System Prompt Architecture

**Without documents** (Normal mode):
```python
system_prompt = build_base_prompt(perspective, audience)
# Uses general knowledge, helpful and informative
```

**With documents** (RAG mode):
```python
system_prompt = build_base_prompt(perspective, audience)
+ strict_rag_instructions()
# Constrains to document content only
```

### Query Structure

Both the chat input and the template form route through a single function,
`rag_core.pipeline.answer()`, so their grounding behaviour cannot drift apart.

```python
# 1. Transform the question (optional stages, off by default)
search_query = transformer.transform(question, chat_history, settings)

# 2. Search both indexes and fuse the rankings
hits = store.search(search_query, dense_weight=0.5)   # vectors + BM25 via RRF

# 3. Optionally re-score the survivors against the question
hits = reranker(search_query, hits, top_k=5)

# 4. Pack them to a budget derived from num_ctx -- never overflow it
context = assemble_context(hits, num_ctx=num_ctx)

# 5. Grounding rules live in the SYSTEM prompt; the user turn carries evidence
answer = llm.generate(
    system_prompt=build_system_prompt(perspective, audience, grounding_mode="strict"),
    user_query=f"DOCUMENT EXCERPTS:\n{context.text}\n\nQUESTION: {question}",
)

# 6. Verify the citations rather than trusting them
report = verify_citations(answer, context.sources)
```

Two things changed from earlier versions and both matter:

**The whole document is no longer sent.** Only retrieved passages are, packed to
a measured budget. Previously the entire text was pasted in and anything beyond
the context limit was silently truncated -- so on a long document the model was
answering from a fragment without anyone knowing.

**Grounding rules moved to the system prompt.** They used to be restated in every
user message, where they competed with the conversation around them and weakened
as history grew. In the system prompt they stay in force for the whole exchange.

---

## Troubleshooting

### "The document doesn't mention this" (but you think it does)

**Possible causes**:
1. Information might be in an image/graph (not extracted)
2. Information might be implied but not explicitly stated
3. Text extraction might have missed a section

**Solution**, in order of usefulness:
1. Open **"🔍 Inspect Indexed Chunks"** and confirm the text was extracted at
   all. Images, graphs and scanned pages yield nothing.
2. Open **"🔬 Retrieval Diagnostics"** on the answer to see what was actually
   searched for and how many passages came back. This separates a *retrieval*
   failure from a *generation* failure.
3. Slide **semantic ↔ keyword balance** toward keyword if you are searching for
   an exact term, or toward semantic if you are paraphrasing.
4. Turn on **"Search multiple phrasings"** or **"Re-rank retrieved passages"**
   under 🧪 Advanced Techniques.
5. Raise **"Passages sent to the model"** in 🎛️ Retrieval Tuning.

### AI still seems to use external knowledge

**Check**:
1. Is the green "RAG MODE ACTIVE" banner showing?
2. Did you process documents after uploading?
3. Are documents still loaded? (Check document list)

**If issue persists**:
- Clear and re-upload documents
- Try a more explicit question
- Ask: "What EXACTLY does the document say about X?"

### AI is too restrictive

**This is intentional** for document fidelity. If you want external context:
- Explicitly request it: "Also provide broader context from your knowledge"
- Or ask separate questions: one for document, one for general info

---

## Comparison Table

| Aspect | Normal Mode (No Docs) | RAG Mode (Docs Uploaded) |
|--------|----------------------|--------------------------|
| **Information Source** | LLM training data | Uploaded documents only |
| **Citations** | General references | Specific page/section numbers |
| **Speculation** | Allowed | Prohibited |
| **External Context** | Provided freely | Only when explicitly requested |
| **Unknown Info** | Will explain generally | States "not in document" |
| **Quotes** | Rare | Frequent and marked |
| **Visual Indicator** | None | Green "RAG MODE ACTIVE" banner |

---

## Best Practices

### 1. Verify RAG Mode is Active
- Look for green banner
- Check document count in caption

### 2. Ask Document-Specific Questions
- ✅ "What does the document say..."
- ✅ "According to this paper..."
- ✅ "Is X mentioned in the document?"

### 3. Use Document Analysis Templates
- Pre-written to enforce document focus
- Already optimized for RAG responses

### 4. Preview Extracted Content
- Enable "Preview Document Content"
- Verify what text was extracted
- Adjust questions based on available content

### 5. Request External Context Explicitly
- When you want broader context, ask for it
- Be clear about when to use external knowledge

---

## Summary

🎯 **Strict RAG Mode Ensures**:
- Responses based ONLY on your documents
- Clear citations to source material
- No hallucinations or fabricated details
- Honest acknowledgment of information gaps
- Faithful representation of document content

📄 **Visual Confirmation**:
- Green "RAG MODE ACTIVE" banner when documents loaded
- Document count in status bar

✅ **Result**:
- Trustworthy, verifiable document analysis
- Perfect for research, study, and professional work
- AI acts as document reader, not general assistant

---

**Enjoy accurate, document-focused analysis!** 📚✨
