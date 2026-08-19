"""
Document Processor for PDF, DOCX and PPTX Files

Extracts text from uploaded documents as *structured blocks* -- units that
retain their page/slide number and section heading -- rather than as one flat
string. Chunking downstream splits within block boundaries, so a chunk never
straddles two pages and every chunk can cite where it came from.

A flat-text rendering is still produced (`blocks_to_text`) for previewing and
for the pre-retrieval code paths.
"""

import io
import os
import re
import tempfile
from typing import Dict, List, Optional

# A block is a dict with keys:
#   text        (str)            the block's text content
#   page        (Optional[int])  1-based page number, PDFs only
#   slide       (Optional[int])  1-based slide number, PPTX only
#   section     (Optional[str])  nearest preceding heading, if detected
#   block_index (int)            position within the document
Block = Dict[str, object]

# Headings commonly found in scientific and technical documents. Used to
# recognize a section even when formatting information has been lost, which is
# the normal case for PDF text extraction.
_KNOWN_SECTIONS = {
    "abstract", "introduction", "background", "related work", "methods",
    "materials and methods", "methodology", "experimental", "results",
    "results and discussion", "discussion", "conclusion", "conclusions",
    "references", "bibliography", "acknowledgments", "acknowledgements",
    "appendix", "supplementary material", "supporting information",
    "limitations", "future work", "data availability",
}

# "3", "3.1", "3.1.2" optionally followed by a period, then a word.
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+\S")


class DocumentProcessor:
    """
    Processes uploaded documents (PDF, DOCX, PPTX) to extract structured text.
    Supports both single file and batch processing.
    """

    # Maximum file size (200 MB)
    MAX_FILE_SIZE = 200 * 1024 * 1024

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx'}

    # Below this ratio of characters per page, a PDF is almost certainly scanned
    # images rather than text, and needs OCR that this app does not perform.
    MIN_CHARS_PER_PAGE = 50

    @staticmethod
    def is_supported_file(filename: str) -> bool:
        """
        Check if file type is supported.

        Args:
            filename: Name of the file

        Returns:
            True if file type is supported, False otherwise
        """
        ext = os.path.splitext(filename.lower())[1]
        return ext in DocumentProcessor.SUPPORTED_EXTENSIONS

    # -------------------------------------------------------------------------
    # Heading detection
    # -------------------------------------------------------------------------
    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        """
        Decide whether a bare line of text is a section heading.

        Kept deliberately conservative: a missed heading only costs some
        metadata precision, while an over-eager rule would label body sentences
        as sections and make metadata filtering actively misleading.
        """
        stripped = line.strip()

        if not (3 <= len(stripped) <= 80):
            return False

        # Body text ends in sentence punctuation; headings generally do not.
        if stripped[-1] in ".,;":
            # Tolerate a trailing period only on numbered headings ("3. Methods").
            if not _NUMBERED_HEADING.match(stripped):
                return False

        normalized = stripped.lower().strip(" :.-")

        if normalized in _KNOWN_SECTIONS:
            return True

        word_count = len(stripped.split())

        if _NUMBERED_HEADING.match(stripped) and word_count <= 10:
            return True

        # ALL CAPS short lines are headings in most extracted layouts.
        if stripped.isupper() and word_count <= 10:
            return True

        return False

    @staticmethod
    def _split_into_sections(text: str, start_section: Optional[str] = None):
        """
        Split raw text into (section_name, body) pairs on detected headings.

        Args:
            text: Raw extracted text.
            start_section: Section carried over from the previous page.

        Returns:
            List of (section_name, body_text) tuples; body may be empty.
        """
        sections = []
        current_section = start_section
        buffer: List[str] = []

        for line in text.split("\n"):
            if DocumentProcessor._looks_like_heading(line):
                if buffer:
                    sections.append((current_section, "\n".join(buffer).strip()))
                    buffer = []
                current_section = line.strip().strip(":")
            else:
                buffer.append(line)

        if buffer:
            sections.append((current_section, "\n".join(buffer).strip()))

        return sections or [(current_section, "")]

    # -------------------------------------------------------------------------
    # Block extraction
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_blocks_from_pdf(file_content: bytes) -> List[Block]:
        """
        Extract structured blocks from a PDF, one or more per page.

        Args:
            file_content: Binary content of the PDF file

        Returns:
            List of blocks carrying page number and detected section
        """
        import PyPDF2

        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        blocks: List[Block] = []
        current_section: Optional[str] = None

        for page_num, page in enumerate(pdf_reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                # One malformed page should not abandon the whole document.
                blocks.append({
                    "text": f"[Error extracting text from page {page_num}: {e}]",
                    "page": page_num,
                    "slide": None,
                    "section": current_section,
                    "block_index": len(blocks),
                })
                continue

            if not text.strip():
                continue

            for section, body in DocumentProcessor._split_into_sections(text, current_section):
                current_section = section
                if body.strip():
                    blocks.append({
                        "text": body.strip(),
                        "page": page_num,
                        "slide": None,
                        "section": section,
                        "block_index": len(blocks),
                    })

        return blocks

    @staticmethod
    def extract_blocks_from_docx(file_content: bytes) -> List[Block]:
        """
        Extract structured blocks from a DOCX file.

        Paragraph styles give real heading information here, so sections are
        detected from the document model rather than guessed from line shape.

        Args:
            file_content: Binary content of the DOCX file

        Returns:
            List of blocks carrying detected section
        """
        import docx

        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name

        try:
            doc = docx.Document(tmp_path)
            blocks: List[Block] = []
            current_section: Optional[str] = None
            paragraph_buffer: List[str] = []

            def flush():
                if paragraph_buffer:
                    blocks.append({
                        "text": "\n\n".join(paragraph_buffer).strip(),
                        "page": None,
                        "slide": None,
                        "section": current_section,
                        "block_index": len(blocks),
                    })
                    paragraph_buffer.clear()

            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue

                style_name = ""
                try:
                    style_name = para.style.name or ""
                except Exception:
                    pass

                is_heading = style_name.startswith("Heading") or style_name == "Title"

                if is_heading or DocumentProcessor._looks_like_heading(text):
                    flush()
                    current_section = text
                else:
                    paragraph_buffer.append(text)

            flush()

            # Tables are extracted as their own blocks so a chunk boundary never
            # lands in the middle of a row.
            for table in doc.tables:
                rows = []
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip(" |"):
                        rows.append(row_text)
                if rows:
                    blocks.append({
                        "text": "[Table]\n" + "\n".join(rows),
                        "page": None,
                        "slide": None,
                        "section": current_section,
                        "block_index": len(blocks),
                    })

            return blocks

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def extract_blocks_from_pptx(file_content: bytes) -> List[Block]:
        """
        Extract structured blocks from a PPTX file, one per slide.

        The slide title becomes the section, which makes slide decks
        filterable by topic.

        Args:
            file_content: Binary content of the PPTX file

        Returns:
            List of blocks carrying slide number and title
        """
        from pptx import Presentation

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pptx') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name

        try:
            prs = Presentation(tmp_path)
            blocks: List[Block] = []

            for slide_num, slide in enumerate(prs.slides, start=1):
                title = None
                try:
                    if slide.shapes.title and slide.shapes.title.text.strip():
                        title = slide.shapes.title.text.strip()
                except Exception:
                    pass

                body_texts = []
                for shape in slide.shapes:
                    if not shape.has_text_frame:
                        continue
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        # Skip the title placeholder; it is already the section.
                        if text and text != title:
                            body_texts.append(text)

                if not body_texts and not title:
                    continue

                blocks.append({
                    "text": "\n".join(body_texts) if body_texts else (title or ""),
                    "page": None,
                    "slide": slide_num,
                    "section": title,
                    "block_index": len(blocks),
                })

            return blocks

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # -------------------------------------------------------------------------
    # Flat-text rendering (preview, export, pre-retrieval paths)
    # -------------------------------------------------------------------------
    @staticmethod
    def blocks_to_text(blocks: List[Block]) -> str:
        """
        Render blocks back to readable flat text with location markers.

        Args:
            blocks: Blocks from any of the extractors

        Returns:
            Human-readable text
        """
        parts = []
        last_marker = None

        for block in blocks:
            if block.get("page") is not None:
                marker = f"--- Page {block['page']} ---"
            elif block.get("slide") is not None:
                marker = f"--- Slide {block['slide']} ---"
            else:
                marker = None

            header = ""
            if marker and marker != last_marker:
                header = marker + "\n"
                last_marker = marker

            section = block.get("section")
            if section:
                header += f"[{section}]\n"

            parts.append(header + str(block.get("text", "")))

        return "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # Legacy flat-text extractors (kept so existing call sites keep working)
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file as flat text."""
        try:
            blocks = DocumentProcessor.extract_blocks_from_pdf(file_content)
            return DocumentProcessor.blocks_to_text(blocks) or "[No text could be extracted from PDF]"
        except ImportError:
            return "[Error: PyPDF2 not installed. Run: pip install PyPDF2]"
        except Exception as e:
            return f"[Error processing PDF: {str(e)}]"

    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from DOCX file as flat text."""
        try:
            blocks = DocumentProcessor.extract_blocks_from_docx(file_content)
            return DocumentProcessor.blocks_to_text(blocks) or "[No text found in DOCX]"
        except ImportError:
            return "[Error: python-docx not installed. Run: pip install python-docx]"
        except Exception as e:
            return f"[Error processing DOCX: {str(e)}]"

    @staticmethod
    def extract_text_from_pptx(file_content: bytes) -> str:
        """Extract text from PPTX file as flat text."""
        try:
            blocks = DocumentProcessor.extract_blocks_from_pptx(file_content)
            return DocumentProcessor.blocks_to_text(blocks) or "[No text found in PPTX]"
        except ImportError:
            return "[Error: python-pptx not installed. Run: pip install python-pptx]"
        except Exception as e:
            return f"[Error processing PPTX: {str(e)}]"

    # -------------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------------
    @staticmethod
    def process_document(uploaded_file) -> Dict:
        """
        Process an uploaded document and extract structured text.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Dictionary with 'filename', 'blocks', 'content', 'error',
            'file_type', 'size' and 'warning'
        """
        result: Dict = {
            'filename': uploaded_file.name,
            'blocks': [],
            'content': '',
            'error': None,
            'warning': None,
            'file_type': None,
            'size': uploaded_file.size,
        }

        if uploaded_file.size > DocumentProcessor.MAX_FILE_SIZE:
            result['error'] = (
                f"File too large ({uploaded_file.size / (1024*1024):.1f} MB). "
                f"Maximum size is 200 MB."
            )
            return result

        ext = os.path.splitext(uploaded_file.name.lower())[1]
        result['file_type'] = ext

        if not DocumentProcessor.is_supported_file(uploaded_file.name):
            result['error'] = f"Unsupported file type: {ext}. Supported types: PDF, DOCX, PPTX"
            return result

        try:
            file_content = uploaded_file.read()
        except Exception as e:
            result['error'] = f"Error reading file: {str(e)}"
            return result

        try:
            if ext == '.pdf':
                result['blocks'] = DocumentProcessor.extract_blocks_from_pdf(file_content)
                result['warning'] = DocumentProcessor._check_pdf_text_yield(
                    file_content, result['blocks']
                )
            elif ext in ('.docx', '.doc'):
                result['blocks'] = DocumentProcessor.extract_blocks_from_docx(file_content)
            elif ext == '.pptx':
                result['blocks'] = DocumentProcessor.extract_blocks_from_pptx(file_content)
            else:
                result['error'] = f"Unsupported file type: {ext}"
                return result

            result['content'] = DocumentProcessor.blocks_to_text(result['blocks'])

            if not result['content'].strip():
                result['error'] = (
                    f"No text could be extracted from {uploaded_file.name}. "
                    f"If this is a scanned document, it needs OCR before upload."
                )

        except ImportError as e:
            result['error'] = f"Missing dependency: {e}. Run: pip install -r requirements.txt"
        except Exception as e:
            result['error'] = f"Error processing document: {str(e)}"

        return result

    @staticmethod
    def _check_pdf_text_yield(file_content: bytes, blocks: List[Block]) -> Optional[str]:
        """
        Warn when a PDF yields suspiciously little text.

        A scanned PDF extracts as empty or near-empty without raising, which
        previously surfaced as an unexplained "not found in the document"
        answer. Naming the cause up front is far more useful.
        """
        try:
            import PyPDF2
            page_count = len(PyPDF2.PdfReader(io.BytesIO(file_content)).pages)
        except Exception:
            return None

        if page_count == 0:
            return None

        total_chars = sum(len(str(b.get("text", ""))) for b in blocks)
        if total_chars / page_count < DocumentProcessor.MIN_CHARS_PER_PAGE:
            return (
                f"Only {total_chars:,} characters extracted from {page_count} pages. "
                f"This PDF is likely scanned images; it needs OCR to be searchable."
            )
        return None

    @staticmethod
    def process_multiple_documents(uploaded_files: List) -> List[Dict]:
        """
        Process multiple uploaded documents.

        Args:
            uploaded_files: List of Streamlit UploadedFile objects

        Returns:
            List of result dictionaries
        """
        return [DocumentProcessor.process_document(f) for f in uploaded_files]

    @staticmethod
    def format_document_content(results: List[Dict],
                               include_filenames: bool = True,
                               separator: str = "\n\n" + "="*80 + "\n\n") -> str:
        """
        Format extracted document content for inclusion in queries.

        Args:
            results: List of processing results from process_multiple_documents
            include_filenames: Whether to include filename headers
            separator: Separator between documents

        Returns:
            Formatted text string ready for LLM input
        """
        formatted_parts = []

        for result in results:
            if result['error']:
                if include_filenames:
                    formatted_parts.append(
                        f"[File: {result['filename']}]\n"
                        f"Error: {result['error']}"
                    )
            elif result['content']:
                if include_filenames:
                    formatted_parts.append(
                        f"[Document: {result['filename']}]\n\n"
                        f"{result['content']}"
                    )
                else:
                    formatted_parts.append(result['content'])

        return separator.join(formatted_parts)

    @staticmethod
    def get_document_summary(result: Dict) -> str:
        """
        Get a brief summary of document processing result.

        Args:
            result: Processing result dictionary

        Returns:
            Human-readable summary string
        """
        if result['error']:
            return f"❌ {result['filename']}: {result['error']}"

        content_length = len(result['content'])
        word_count = len(result['content'].split())
        size_mb = result['size'] / (1024 * 1024)
        block_count = len(result.get('blocks', []))

        return (f"✅ {result['filename']} "
                f"({size_mb:.2f} MB, ~{word_count:,} words, "
                f"{content_length:,} characters, {block_count} blocks)")
