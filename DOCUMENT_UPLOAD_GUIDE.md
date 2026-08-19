# 📄 Document Upload Feature Guide

## Overview

The Local LLM Scientific Assistant supports **uploading, indexing and searching
PDF, DOCX and PPTX documents**. Upload research papers, reports, manuscripts or
slide decks, and ask questions answered from their contents with citations.

---

## Supported File Types

- ✅ **PDF** (.pdf) - Extracts text page by page, detecting section headings
- ✅ **DOCX** (.docx) - Extracts paragraphs and tables, using real heading styles
- ✅ **DOC** (.doc) - Legacy Word format support
- ✅ **PPTX** (.pptx) - Extracts each slide, using slide titles as sections

**Maximum file size**: 200 MB per file
**Multiple files**: Upload several documents at once for comparison or combined analysis

> **Scanned PDFs will not work.** If a PDF contains images of text rather than
> text, there is nothing to extract. The app detects this and tells you at
> upload time; run the file through OCR first.

---

## How Retrieval Works

Understanding this makes the rest of the guide make sense.

```
Your document
   ↓  extracted into blocks that keep their page/slide and section
   ↓  split into chunks (~400 words) that never straddle a page
   ↓  each chunk embedded into a vector AND indexed for keyword search
   ↓
Your question
   ↓  searched two ways at once:
   ↓     • semantically -- finds passages that mean the same thing
   ↓     • by keyword   -- finds exact terms, part numbers, gene names
   ↓  the two rankings are merged
   ↓  the best passages are packed into the model's context window
   ↓
Answer, citing [S1] [S2] traced back to real pages
```

**Why both search methods?** Each fails where the other succeeds. Ask "how warm
were the cells kept" and semantic search finds "cultured at 37 degrees" despite
sharing no words with your question. Ask for catalog number "ZZQX7-4419" and
keyword search finds it exactly, where semantic search would return
vaguely-similar passages. Running both is what makes the app reliable across
question types.

**Nothing is written to disk.** The index lives in memory for the session only.
Close the app, or press **"🗑️ Clear Vector Store"**, and it is gone.

---

## How to Use

### Basic Workflow

1. **Navigate to Chat page**
2. **Click "📄 Upload Documents (PDF, DOCX, PPTX)"** expander
3. **Choose your files** using the file uploader
4. **Click "📥 Process Documents"** to extract, chunk and index them
5. **Ask questions** or use templates -- retrieval runs automatically

### Detailed Steps

#### Step 1: Upload Documents

```
1. Click the "📄 Upload Documents" expander section
2. Click "Browse files" or drag-and-drop
3. Select one or more PDF/DOCX files
4. Files appear in the uploader list
```

#### Step 2: Process Documents

```
1. Click the "📥 Process Documents" button
2. Text is extracted, split into chunks, and embedded
   (a progress bar shows the embedding stage)
3. See success messages with document stats:
   ✅ filename.pdf (2.5 MB, ~5,234 words, 32,145 characters, 47 blocks)
   ✅ Indexed 63 chunks -- documents are now searchable.
```

The sidebar then shows **🧠 Vector store: 63 chunks from 1 document(s)**.

#### Step 3: Ask Questions

**Option A - Direct Questions:**
```
Type in chat: "What are the main findings of this study?"
The app searches your documents and supplies the most relevant
passages to the model, cited as [S1], [S2], ...
```

**Option B - Use Templates:**
```
1. Open "Question Templates" expander
2. Filter to "Document Analysis" category
3. Select template like "Summarize Uploaded Document"
4. Submit - retrieval runs exactly as it does for typed questions
```

**Option C - Custom Analysis:**
```
Type any question referencing "the document" or "this paper"
Examples:
- "What methodology did they use?"
- "Compare the findings in these papers"
- "What are the limitations mentioned?"
```

---

## Document Management

### View Loaded Documents

After processing, the expander shows:
- ✅ Filename and word count
- 🗑️ Remove button for each document
- 🗑️ Clear All button

### Inspect Indexed Chunks

The **"🔍 Inspect Indexed Chunks"** expander shows exactly what retrieval
searches over: every chunk, its source page or slide, its detected section, and
its size. Worth a look when an answer misses something you know is in the
document -- it tells you whether the text was extracted and split sensibly, or
whether the problem is further down the pipeline.

### Preview Content

Check the **"👁️ Preview Document Content"** box to see:
- First 2,000 characters of extracted text
- Total character count
- Formatted text with page markers (PDF) or sections (DOCX)

### Remove Documents

**Remove one document:**
- Click 🗑️ next to the filename

**Remove all documents:**
- Click "🗑️ Clear All Documents" button

---

## Document Analysis Templates

The app includes **8 specialized templates** for document work:

### 1. Summarize Uploaded Document
**Purpose**: Get comprehensive summary with key findings  
**Use when**: You want an overview of the document as a whole

### 2. Extract Key Points from Document
**Purpose**: Bullet-point list of main takeaways  
**Use when**: You need quick reference points

### 3. Analyze Methodology in Document
**Purpose**: Deep dive into research methods  
**Use when**: Evaluating study design and approach

### 4. Compare Multiple Documents
**Purpose**: Side-by-side analysis of uploaded papers  
**Use when**: You uploaded 2+ documents for comparison

### 5. Answer Question from Document
**Purpose**: Specific Q&A about document content  
**Use when**: You have targeted questions

### 6. Create Outline from Document
**Purpose**: Hierarchical structure of document  
**Use when**: You need to understand organization

### 7. Identify Knowledge Gaps
**Purpose**: Find limitations and future research needs  
**Use when**: Planning follow-up research

### 8. Simplify Technical Document
**Purpose**: Translate complex content to accessible language  
**Use when**: Making technical papers understandable

---

## Use Cases

### For Researchers

**Literature Review:**
```
1. Upload 5-10 related papers
2. Use "Compare Multiple Documents" template
3. Get synthesis of findings and methodologies
```

**Paper Analysis:**
```
1. Upload a target paper
2. Use "Analyze Methodology" template
3. Ask follow-up questions about specific sections
```

**Gap Analysis:**
```
1. Upload current state-of-field papers
2. Use "Identify Knowledge Gaps" template
3. Plan your research contribution
```

### For Students

**Study Papers:**
```
1. Upload assigned reading (PDF textbook chapter)
2. Use "Extract Key Points" template
3. Ask questions: "Explain the concept of [X] from this reading"
```

**Essay Research:**
```
1. Upload 3-4 source papers
2. Ask: "What do these sources say about [topic]?"
3. Get synthesized information for your essay
```

### For Educators

**Lesson Planning:**
```
1. Upload research paper on teaching topic
2. Use "Simplify Technical Document" template
3. Get accessible explanations for students
```

**Assignment Creation:**
```
1. Upload source material
2. Ask: "Create 5 discussion questions about this paper"
3. Use for class activities
```

---

## Tips for Best Results

### 📝 Document Preparation

1. **Use text-based PDFs**: Scanned PDFs contain images, not text, and extract
   to nothing. The app detects this and warns you at upload; run the file
   through OCR first.
2. **File size**: Up to 200 MB per file. Larger files simply take longer to
   embed at upload — they do *not* degrade answer quality, because retrieval
   searches for the relevant passages rather than sending everything.
3. **Clean formatting**: Real heading styles (in DOCX) and consistent section
   headings (in PDF) let the app tag chunks with their section, which makes
   metadata filtering more accurate.
4. **Multiple files**: Upload all related docs together for comparison — you can
   restrict any search to a subset later from 🎛️ Retrieval Tuning.

### 🎯 Querying Tips

1. **Be specific**: "What methodology is used?" vs "Tell me about this"
2. **Reference sections**: "Summarize the introduction" works well
3. **Multiple docs**: Say "Compare paper 1 and paper 2" or "across all documents"
4. **Follow-up**: Ask clarifying questions in the same conversation

### ⚙️ Performance Tips

1. **Context window**: *Raise* "Context Length", never lower it, when documents
   are loaded. Retrieved passages are packed to a budget derived from that
   setting, so lowering it means less evidence reaches the model. The app never
   overflows the window -- if passages will not fit it says how many it left
   out, in the 🔬 Retrieval Diagnostics panel.
2. **Processing time**: The slow step is embedding, not reading. A 100-page
   document is typically a few seconds; the "Add contextual headers" option is
   far slower because it calls the model once per chunk.
3. **Multiple queries**: Documents stay indexed for the session - ask many
   questions without re-processing.
4. **Document size is no longer a limit on answer quality.** Retrieval finds the
   relevant passages wherever they are, so a 500-page manual works as well as a
   2-page memo. Earlier versions pasted whole documents into the prompt and
   silently lost everything past the context limit.

---

## Troubleshooting

### ❌ "Error: PyPDF2 not installed"

**Solution:**
```bash
pip install PyPDF2
```

### ❌ "Error: python-docx not installed"

**Solution:**
```bash
pip install python-docx
```

### ❌ "No embedding model found" / "Documents extracted but NOT indexed"

Document *search* needs an embedding model, which is separate from your chat
model.

**Solution:**
```bash
ollama pull nomic-embed-text
```
Then process the documents again. Until you do, the app answers from general
knowledge and says so plainly -- it will not pretend to be reading your files.

### ❌ "This PDF is likely scanned images; it needs OCR to be searchable"

The PDF holds pictures of text rather than text. Nothing in this app can read
it. Run it through an OCR tool first (macOS Preview, Adobe Acrobat, or
`ocrmypdf`), then upload the OCR'd version.

### ❌ Answers cite a source number that does not exist

The app detects this and shows a warning above the answer. It means the model
invented a citation. Lower the temperature, or switch to a larger chat model.

### ❌ "File too large (215.2 MB). Maximum size is 200 MB"

**Solutions:**
- Split the PDF into smaller files and upload them together
- Compress the PDF
- Extract only the relevant pages

### ❌ "No text could be extracted from PDF"

**Causes & Solutions:**
- **Scanned PDF**: Use OCR software first (Adobe Acrobat, online tools)
- **Image-based PDF**: Convert to text-based PDF
- **Encrypted PDF**: Remove password protection first
- **Corrupted file**: Try re-downloading the PDF

### ❌ Garbled or strange characters

**Solutions:**
- Check PDF encoding (should be UTF-8)
- Some PDFs have encoding issues - try different version
- Use "Preview Content" to see what was extracted

### ⚠️ Processing very slow

**Solutions:**
- Reduce document size
- Upload fewer documents at once
- Check file isn't password-protected

---

## Technical Details

### Text Extraction

**PDF Processing:**
- Uses PyPDF2 library
- Extracts text page-by-page
- Preserves page markers: `--- Page 1 ---`
- Handles multi-column layouts (best effort)

**DOCX Processing:**
- Uses python-docx library
- Extracts paragraph text
- Extracts table content with formatting
- Preserves basic structure

### Content Formatting

Documents are formatted as:
```
[Document: filename.pdf]

--- Page 1 ---
[extracted text]

--- Page 2 ---
[extracted text]

=====================================

[Document: another.docx]

[extracted text]
[Table]
header1 | header2
data1   | data2
```

### Integration with Queries

When you submit a query with documents loaded:
```
Your query text

--- UPLOADED DOCUMENT CONTENT ---

[formatted document text]
```

This combined text goes to the LLM, so it has full context.

---

## Limitations

1. **Text-only extraction**: Images, graphs, and equations are not extracted
2. **OCR not included**: Scanned PDFs require pre-processing
3. **Layout preserved minimally**: Complex formatting may be lost
4. **Token limits**: Very large documents may exceed context window
5. **No PDF editing**: Extract text only, cannot modify source files

---

## Advanced Usage

### Batch Processing

```python
# Upload multiple papers at once
# Process all together
# Ask: "Create a comparison table of methodologies across all papers"
```

### Iterative Analysis

```
1. Upload document
2. Get summary (Template: "Summarize Document")
3. Ask follow-up: "Explain the statistical analysis in detail"
4. Ask more: "What are the limitations of this approach?"
5. Finally: "How could this be improved?"
```

### Teaching Applications

```
# Scenario: Explaining complex paper to students
1. Upload research paper (PDF)
2. Set Perspective: "High School Teacher"
3. Set Audience: "High School Student"  
4. Use: "Simplify Technical Document" template
5. Get: Student-appropriate explanation
```

---

## FAQ

**Q: Can I upload images of text?**  
A: No, only text-based PDFs work. Use OCR software first.

**Q: How many documents can I upload?**  
A: No hard limit, but more documents = larger context = slower processing

**Q: Do documents persist between sessions?**  
A: No, documents are session-only. Re-upload after browser refresh.

**Q: Can I edit the extracted text?**  
A: Not directly, but you can copy the preview and paste into your query.

**Q: Does it work with non-English documents?**  
A: Yes, if the PDF/DOCX uses UTF-8 encoding.

**Q: Can I upload .txt files?**  
A: Not currently, but you can copy-paste text directly into queries.

---

## Next Steps

1. ✅ Install document libraries: `pip install PyPDF2 python-docx`
2. ✅ Try uploading a sample PDF
3. ✅ Experiment with document analysis templates
4. ✅ Ask questions about your uploaded documents
5. ✅ Combine with perspective/audience settings for targeted analysis

**Happy analyzing! 📚**
