"""
HTML Export Utilities
Handles conversion of chat conversations to formatted HTML documents with Bootstrap styling.
Supports Markdown rendering including tables.
"""

import html
import re
import datetime
from typing import List

# -----------------------------
# Utility: HTML + Markdown helpers
# -----------------------------
def _escape_with_breaks(text: str) -> str:
    """
    Escape HTML and convert newlines to <br> for readable HTML output.
    
    Args:
        text: Raw text to escape
        
    Returns:
        HTML-safe text with line breaks
    """
    if text is None:
        return ""
    return html.escape(str(text)).replace("\n", "<br>")


def _postprocess_tables_for_bootstrap(html_in: str) -> str:
    """
    Add Bootstrap classes to <table> and wrap in a responsive container.
    Ensures clean rendering in exported HTML.
    
    Args:
        html_in: HTML string potentially containing tables
        
    Returns:
        HTML with Bootstrap-styled tables
    """
    if not html_in:
        return html_in

    # Wrap each <table> ... </table> block with a responsive div and add classes.
    def _add_classes(match):
        table_block = match.group(0)
        # Add classes to the opening <table> tag (avoid duplicating)
        table_block = re.sub(
            r"<table(?![^>]*\bclass=)",
            '<table class="table table-striped table-bordered table-sm"',
            table_block,
            flags=re.IGNORECASE
        )
        # If there's already a class, append ours
        table_block = re.sub(
            r'<table([^>]*\bclass=")([^"]*)(")',
            lambda m: f'<table{m.group(1)}{m.group(2)} table table-striped table-bordered table-sm{m.group(3)}',
            table_block,
            flags=re.IGNORECASE
        )
        return f'<div class="table-responsive">{table_block}</div>'

    return re.sub(r"<table[\s\S]*?</table>", _add_classes, html_in, flags=re.IGNORECASE)


def _markdown_to_html(text: str) -> str:
    """
    Convert Markdown (including GitHub-style pipe tables) to HTML.
    Tries to use 'markdown' package if available (with 'tables' extension).
    Falls back to a very small pipe-table converter if not installed.
    
    Args:
        text: Markdown-formatted text
        
    Returns:
        HTML representation of the markdown
    """
    if text is None:
        return ""

    # Fast path: if there's no '|' looking like a table and no markdown markers,
    # we can just escape + <br>.
    looks_like_table = bool(re.search(r"^\s*\|.+\|\s*$", text, flags=re.MULTILINE))
    looks_like_md = looks_like_table or ("**" in text or "__" in text or "#" in text or "`" in text or "-" in text)

    try:
        # Use Python-Markdown if available
        import markdown as md  # type: ignore
        html_out = md.markdown(
            text,
            extensions=["tables", "fenced_code", "nl2br"]
        )
        return _postprocess_tables_for_bootstrap(html_out)
    except Exception:
        # No markdown package or it failed; use a very small fallback
        if not looks_like_md:
            return _escape_with_breaks(text)

        # Fallback: convert pipe tables only; pass rest escaped with <br>.
        lines = text.splitlines()
        html_lines: List[str] = []
        i = 0
        while i < len(lines):
            # Detect a pipe-table block starting at i
            if re.match(r"^\s*\|.*\|\s*$", lines[i] or ""):
                block = [lines[i]]
                i += 1
                while i < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[i] or ""):
                    block.append(lines[i])
                    i += 1
                html_lines.append(_pipe_table_block_to_html(block))
            else:
                html_lines.append(_escape_with_breaks(lines[i]))
                i += 1

        joined = "\n".join(html_lines)
        return _postprocess_tables_for_bootstrap(joined)


def _pipe_table_block_to_html(block_lines: List[str]) -> str:
    """
    Minimal GitHub-style pipe table -> HTML.
    Expects something like:
      | h1 | h2 |
      | --- | ---: |
      | a | b |
    Alignments are detected but only left/center/right applied to <th>/<td>.
    
    Args:
        block_lines: List of table lines in markdown format
        
    Returns:
        HTML table representation
    """
    # Clean and split cells
    rows = [re.split(r"\s*\|\s*", ln.strip().strip("|")) for ln in block_lines]
    if len(rows) < 2:
        # Not a valid table, escape
        return _escape_with_breaks("\n".join(block_lines))

    header = rows[0]
    divider = rows[1]
    body = rows[2:] if len(rows) > 2 else []

    # Parse alignment from divider row
    aligns = []
    for cell in divider:
        cell = cell.strip()
        if re.match(r"^:-+:$", cell):
            aligns.append("center")
        elif re.match(r"^-+:$", cell):
            aligns.append("right")
        elif re.match(r"^:-+$", cell) or re.match(r"^-+$", cell):
            aligns.append("left")
        else:
            aligns.append("left")

    def _td(cell: str, idx: int, header: bool = False) -> str:
        tag = "th" if header else "td"
        align = aligns[idx] if idx < len(aligns) else "left"
        style = f' style="text-align:{align};"'
        return f"<{tag}{style}>{html.escape(cell)}</{tag}>"

    # Build HTML
    thead = "<thead><tr>" + "".join(_td(c, i, True) for i, c in enumerate(header)) + "</tr></thead>"
    tbody_rows = []
    for r in body:
        # pad/truncate row to header length for safety
        r2 = (r + [""] * len(header))[:len(header)]
        tds = "".join(_td(c, i, False) for i, c in enumerate(r2))
        tbody_rows.append(f"<tr>{tds}</tr>")
    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def _bootstrap_head(title: str) -> str:
    """
    Generate Bootstrap 5 HTML head section with CDN links.
    
    Args:
        title: Page title
        
    Returns:
        HTML head section
    """
    # If you need offline/no-CDN, replace the <link> with a <style> and paste minified Bootstrap 5 CSS inline.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"
      integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
<style>
/* Minor niceties for transcript readability */
.chat-card {{ margin-bottom: 1rem; }}
.badge-role {{ font-size: 0.75rem; }}
pre, code {{ white-space: pre-wrap; word-wrap: break-word; }}
/* Tables */
table {{ width: 100%; }}
/* Context badges */
.context-info {{ font-size: 0.85rem; margin-top: 0.5rem; }}
</style>
</head>
<body class="bg-light">
"""


def _bootstrap_foot() -> str:
    """
    Generate Bootstrap 5 HTML footer section with CDN scripts.
    
    Returns:
        HTML footer section
    """
    return """
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz"
        crossorigin="anonymous"></script>
</body>
</html>
"""


def _render_rich_html(text: str) -> str:
    """
    Render text as HTML with Markdown (incl. tables) support when possible.
    Falls back to escaped text with <br> on failure.
    
    Args:
        text: Text to render
        
    Returns:
        HTML representation
    """
    try:
        return _markdown_to_html(text)
    except Exception:
        # Absolute safety net
        return _escape_with_breaks(text)


# -----------------------------
# HTML builders
# -----------------------------
def build_pair_html(index: int, user_text: str, ai_text: str, 
                   created_at: str = None, perspective: str = None, 
                   audience: str = None, evaluation_html: str = "") -> str:
    """
    Single Q&A as a standalone HTML document.
    
    Args:
        index: Question index number
        user_text: User's question
        ai_text: AI's response
        created_at: Timestamp (optional)
        perspective: Who wrote the answer (optional)
        audience: Intended audience (optional)
        evaluation_html: Pre-rendered evaluation card, appended after the
            answer (optional; "" when the answer was never evaluated)
        
    Returns:
        Complete HTML document string
    """
    title = f"Conversation Q&A #{index+1}"
    created_at = created_at or datetime.datetime.now().isoformat()

    # Use rich HTML so Markdown tables render as real <table>
    user_html = _render_rich_html(user_text)
    ai_html = _render_rich_html(ai_text)
    
    # Add context info if available
    context_html = ""
    if perspective or audience:
        context_parts = []
        if perspective:
            context_parts.append(f'<span class="badge bg-info">Perspective: {html.escape(perspective)}</span>')
        if audience:
            context_parts.append(f'<span class="badge bg-warning">Audience: {html.escape(audience)}</span>')
        context_html = f'<div class="context-info">{" ".join(context_parts)}</div>'

    body = f"""
{_bootstrap_head(title)}
<div class="container py-4">
  <header class="mb-4">
    <h1 class="h3">Conversation Q&amp;A <span class="text-muted">#{index+1}</span></h1>
    <p class="text-muted mb-0">Exported: {html.escape(created_at)}</p>
    {context_html}
  </header>

  <article class="card chat-card shadow-sm">
    <div class="card-header bg-white">
      <span class="badge text-bg-primary badge-role">You</span>
    </div>
    <div class="card-body">
      <div class="card-text">{user_html}</div>
    </div>
  </article>

  <article class="card chat-card shadow-sm">
    <div class="card-header bg-white">
      <span class="badge text-bg-success badge-role">AI</span>
    </div>
    <div class="card-body">
      <div class="card-text">{ai_html}</div>
    </div>
  </article>
{evaluation_html}
</div>
{_bootstrap_foot()}
"""
    return body


def build_conversation_html(history: list, title: str = "Conversation Transcript",
                           perspective: str = None, audience: str = None,
                           evaluation_html: list = None) -> str:
    """
    Full conversation HTML document.
    
    Args:
        history: List of tuples (user_query, ai_response)
        title: Document title
        perspective: Who wrote the answers (optional)
        audience: Intended audience (optional)
        evaluation_html: Pre-rendered evaluation cards, parallel to
            `history`; "" entries mean that Q&A was not evaluated (optional)
        
    Returns:
        Complete HTML document string
    """
    evaluation_html = evaluation_html or []
    created_at = datetime.datetime.now().isoformat()
    
    # Add context info if available
    context_html = ""
    if perspective or audience:
        context_parts = []
        if perspective:
            context_parts.append(f'<span class="badge bg-info">Perspective: {html.escape(perspective)}</span>')
        if audience:
            context_parts.append(f'<span class="badge bg-warning">Audience: {html.escape(audience)}</span>')
        context_html = f'<div class="context-info">{" ".join(context_parts)}</div>'
    
    cards = []
    for i, (q, a) in enumerate(history):
        eval_card = evaluation_html[i] if i < len(evaluation_html) else ""
        cards.append(f"""
  <section id="qa-{i}" class="mb-4">
    <div class="card chat-card shadow-sm">
      <div class="card-header bg-white d-flex justify-content-between align-items-center">
        <span class="badge text-bg-primary badge-role">You</span>
        <small class="text-muted">#{i+1}</small>
      </div>
      <div class="card-body">
        <div class="card-text">{_render_rich_html(q)}</div>
      </div>
    </div>

    <div class="card chat-card shadow-sm">
      <div class="card-header bg-white">
        <span class="badge text-bg-success badge-role">AI</span>
      </div>
      <div class="card-body">
        <div class="card-text">{_render_rich_html(a)}</div>
      </div>
    </div>
{eval_card}
  </section>
""")

    html_doc = f"""
{_bootstrap_head(title)}
<div class="container py-4">
  <header class="mb-4">
    <h1 class="h3">{html.escape(title)}</h1>
    <p class="text-muted mb-0">Exported: {html.escape(created_at)}</p>
    {context_html}
  </header>

  <nav class="mb-4">
    <div class="list-group">
      {"".join([f'<a class="list-group-item list-group-item-action" href="#qa-{i}">Q&A #{i+1}'
                + (' <span class="badge text-bg-warning ms-1">evaluated</span>' if i < len(evaluation_html) and evaluation_html[i] else '')
                + '</a>' for i in range(len(history))])}
    </div>
  </nav>

  {"".join(cards)}
</div>
{_bootstrap_foot()}
"""
    return html_doc


def safe_filename(stem: str, suffix: str = ".html") -> str:
    """
    Create a safe filename by removing special characters.
    
    Args:
        stem: Base filename
        suffix: File extension (default: .html)
        
    Returns:
        Safe filename string
    """
    stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))
    if not stem:
        stem = "conversation"
    return f"{stem}{suffix}"
