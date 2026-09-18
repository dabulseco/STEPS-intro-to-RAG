"""
Rendering and export for LLM-as-judge evaluations.

Three concerns live here, intentionally kept out of `rag_core.eval` so the
engine stays UI-free and importable from scripts:

1. **Streamlit rendering** -- `render_metric_tile` and `render_evaluation_report`
   draw a single metric's result and a full evaluation report as Streamlit
   widgets, including parse-failure fallback styling.
2. **CSV / Markdown / HTML export** -- `evaluations_to_csv`,
   `evaluations_to_markdown`, `render_evaluations_html` produce the three
   download payloads the sidebar offers. CSV is long-form (one row per
   metric) with full question and answer text per user preference.

The HTML export reuses `_bootstrap_head` / `_bootstrap_foot` /
`_render_rich_html` from `utils.html_export` so the rendered report visually
matches the existing per-Q&A exports the app already produces.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
from typing import Dict, List, Optional, Sequence, Tuple

# Streamlit is only needed for the render_* functions. Importing this module
# from a non-Streamlit script (e.g. a CLI exporter) should still work for
# the CSV / Markdown / HTML paths -- they import lazily below.
import streamlit as st

from rag_core.eval import EvaluationReport, MetricResult, split_eval_key

# Reuse the existing Bootstrap scaffolding so visual style is consistent.
from utils.html_export import _bootstrap_head, _bootstrap_foot, _render_rich_html

# ---------------------------------------------------------------------------
# Metric rendering -- one tile per metric, plus a full report layout
# ---------------------------------------------------------------------------

# Stable display order matches the dispatch order in rag_core.eval.METRICS
# (groundedness -> relevance -> faithfulness -> context_precision ->
# context_recall), which is the notebook's Metric 1..5 ordering.
METRIC_DISPLAY_ORDER: Sequence[str] = (
    "groundedness",
    "relevance",
    "faithfulness",
    "context_precision",
    "context_recall",
)


def _rating_color(rating: Optional[int]) -> str:
    """Map a 1-5 rating to a Bootstrap background color for the tile."""
    if rating is None:
        return "secondary"
    if rating >= 4:
        return "success"
    if rating >= 3:
        return "warning"
    return "danger"


def render_metric_tile(metric: MetricResult) -> None:
    """
    Render one MetricResult as a compact Bootstrap-style tile in Streamlit.

    The tile is laid out as: badge with metric name, big rating number (or
    em-dash on parse failure), and a one-line evaluation summary. Clicking
    the expander reveals the full explanation, steps, and raw judge output.
    """
    color = _rating_color(metric.rating)
    rating_display = f"{metric.rating}/5" if metric.rating is not None else "—"

    st.markdown(
        f"""
<div class="card mb-2" style="border-left: 4px solid var(--bs-{color});">
  <div class="card-body py-2 px-3">
    <div class="d-flex justify-content-between align-items-center">
      <strong>{metric.name.replace('_', ' ').title()}</strong>
      <span class="badge text-bg-{color}">{rating_display}</span>
    </div>
    <div class="small text-muted mt-1">{metric.evaluation or '(no summary)'}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if metric.parse_error:
        st.caption(f"⚠️ Parse error: {metric.parse_error}")
        with st.expander("Raw judge output", expanded=False):
            st.text(metric.raw or "(empty)")
    elif metric.explanation or metric.steps:
        with st.expander("Details", expanded=False):
            if metric.steps:
                st.markdown("**Steps:**")
                st.text(metric.steps)
            if metric.explanation:
                st.markdown("**Explanation:**")
                st.text(metric.explanation)


def render_evaluation_report(report: EvaluationReport) -> None:
    """
    Render a full EvaluationReport -- one tile per metric in display order.

    Cached evaluation reports carry no progress; this is also the
    post-run rendering path used when a fresh evaluation completes.
    """
    st.caption(
        f"Judge: `{report.judge_model}` · "
        f"{report.elapsed_seconds:.1f}s · "
        f"{sum(1 for m in report.metrics.values() if m.parsed)}/"
        f"{len(report.metrics)} parsed"
    )

    # Lay tiles out in a 5-column row on wide screens; fall back to
    # stacked rendering if there are fewer metrics.
    cols = st.columns(min(len(report.metrics), 5))
    for index, name in enumerate(METRIC_DISPLAY_ORDER):
        if name not in report.metrics:
            continue
        with cols[index % len(cols)]:
            render_metric_tile(report.metrics[name])


# ---------------------------------------------------------------------------
# Cache-key lookup helpers -- used by the export functions to recover the
# original question and answer text from the eval_key.
# ---------------------------------------------------------------------------

def _history_lookup(history: Sequence[Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
    """
    Map question_hash -> (question, answer) for the current chat history.

    Mirrors the hash produced by `make_eval_key` so the export can recover
    the question text from `question_hash` alone.
    """
    out: Dict[str, Tuple[str, str]] = {}
    for q, a in history:
        h = hashlib.sha1(f"{q}\x00{a}".encode("utf-8")).hexdigest()[:12]
        out[h] = (q, a)
    return out


# ---------------------------------------------------------------------------
# CSV export -- long-form, full question + answer text per user preference
# ---------------------------------------------------------------------------

# Column order chosen for analyst ergonomics: identifiers first, then the
# metric that was scored, then the rating, then the surrounding text that
# gives the row its meaning. Full text is intentional -- a CSV is for
# analysis, and reading the question again from a separate file is friction.
CSV_COLUMNS: Sequence[str] = (
    "question_hash",
    "judge_model",
    "metric",
    "rating",
    "evaluation",
    "elapsed_seconds",
    "question",
    "answer",
)


def evaluations_to_csv(
    evaluations: Dict[str, EvaluationReport],
    history: Sequence[Tuple[str, str]],
) -> str:
    """
    Long-form CSV: one row per (question, judge_model, metric).

    Rows for the same question share the same `question` and `answer`
    columns. Missing ratings render as empty cells rather than `None`
    so spreadsheet apps treat them correctly. Returns the CSV body as a
    string (no BOM); call `st.download_button` with `data=` directly.
    """
    lookup = _history_lookup(history)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for key, report in evaluations.items():
        try:
            q_hash, judge_model = split_eval_key(key)
        except ValueError:
            # Skip malformed keys -- safer than crashing an export the user
            # just clicked. A future cleanup pass can prune these.
            continue
        question, answer = lookup.get(q_hash, ("", ""))
        for metric_name in METRIC_DISPLAY_ORDER:
            m = report.metrics.get(metric_name)
            if m is None:
                continue
            writer.writerow({
                "question_hash": q_hash,
                "judge_model": judge_model,
                "metric": metric_name,
                "rating": "" if m.rating is None else m.rating,
                "evaluation": m.evaluation,
                "elapsed_seconds": f"{report.elapsed_seconds:.3f}",
                "question": question,
                "answer": answer,
            })

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Markdown export -- readable summary, structured by question
# ---------------------------------------------------------------------------

def evaluations_to_markdown(
    evaluations: Dict[str, EvaluationReport],
    history: Sequence[Tuple[str, str]],
) -> str:
    """
    Markdown export grouped by question, with one table per judge model.

    Suitable for sharing or pasting into a notes app. Questions whose
    hash does not match the current chat history are still included --
    their hash + judge model + ratings are recorded, but the question
    body is omitted (the chat history was likely cleared).
    """
    lookup = _history_lookup(history)
    by_question: Dict[str, Dict[str, EvaluationReport]] = {}
    for key, report in evaluations.items():
        try:
            q_hash, judge_model = split_eval_key(key)
        except ValueError:
            continue
        by_question.setdefault(q_hash, {})[judge_model] = report

    out: List[str] = ["# RAG Evaluation Report", ""]
    if not by_question:
        out.append("_No evaluations to export._")
        return "\n".join(out)

    for q_hash in sorted(by_question.keys()):
        q, a = lookup.get(q_hash, ("", ""))
        if q:
            out.append(f"## Question\n\n> {q}\n")
            out.append(f"## Answer\n\n{a}\n")
        else:
            out.append(f"## Question (hash `{q_hash}`)\n")
            out.append("_Question text not in current chat history._\n")

        for judge_model in sorted(by_question[q_hash].keys()):
            report = by_question[q_hash][judge_model]
            out.append(f"### Judge: `{judge_model}` ({report.elapsed_seconds:.1f}s)\n")
            out.append("| Metric | Rating | Evaluation |")
            out.append("|--------|--------|------------|")
            for name in METRIC_DISPLAY_ORDER:
                m = report.metrics.get(name)
                if m is None:
                    continue
                rating = f"**{m.rating}** / 5" if m.rating is not None else "—"
                eval_text = m.evaluation.replace("|", "\\|") if m.evaluation else ""
                out.append(f"| {name} | {rating} | {eval_text} |")
            out.append("")
        out.append("---\n")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML export -- standalone Bootstrap document, one section per question
# ---------------------------------------------------------------------------

def _rating_badge_html(rating: Optional[int]) -> str:
    color = _rating_color(rating)
    if rating is None:
        return f'<span class="badge text-bg-{color}">no rating</span>'
    return f'<span class="badge text-bg-{color}">{rating} / 5</span>'


def _metric_tile_html(m: MetricResult) -> str:
    """One metric as a Bootstrap card; the explanation folds into <details>."""
    color = _rating_color(m.rating)
    eval_text = html.escape(m.evaluation or "(no summary)")
    extra = ""
    if m.parse_error:
        extra = f'<div class="small text-danger mt-1">Parse error: {html.escape(m.parse_error)}</div>'
    elif m.explanation:
        extra = (
            '<details class="small mt-1"><summary>Explanation</summary>'
            f'<div class="mt-1">{html.escape(m.explanation)}</div></details>'
        )
    return f"""
<div class="col-md-6 col-lg-4 mb-2">
  <div class="card h-100" style="border-left: 4px solid var(--bs-{color});">
    <div class="card-body py-2 px-3">
      <div class="d-flex justify-content-between align-items-center">
        <strong>{m.name.replace('_', ' ').title()}</strong>
        {_rating_badge_html(m.rating)}
      </div>
      <div class="small text-muted mt-1">{eval_text}</div>{extra}
    </div>
  </div>
</div>"""


def _judge_blocks_html(reports_by_judge: Dict[str, EvaluationReport]) -> List[str]:
    """One block of metric tiles per judge model, sorted by judge name."""
    blocks: List[str] = []
    for judge_model in sorted(reports_by_judge.keys()):
        report = reports_by_judge[judge_model]
        tiles = [
            _metric_tile_html(report.metrics[name])
            for name in METRIC_DISPLAY_ORDER
            if name in report.metrics
        ]
        blocks.append(f"""
<section class="mb-3">
  <h6 class="mb-1">Judge: <code>{html.escape(judge_model)}</code>
    <small class="text-muted">({report.elapsed_seconds:.1f}s)</small>
  </h6>
  <div class="row">{''.join(tiles)}</div>
</section>""")
    return blocks


def evaluations_for_pair(
    evaluations: Dict[str, EvaluationReport],
    question: str,
    answer: str,
) -> Dict[str, EvaluationReport]:
    """
    All evaluations recorded for one Q&A pair, keyed by judge model.

    A pair can have several entries when the user re-ran it with different
    judge models; every one of them is returned.
    """
    q_hash = hashlib.sha1(f"{question}\x00{answer}".encode("utf-8")).hexdigest()[:12]
    out: Dict[str, EvaluationReport] = {}
    for key, report in evaluations.items():
        try:
            h, judge_model = split_eval_key(key)
        except ValueError:
            continue
        if h == q_hash:
            out[judge_model] = report
    return out


def render_pair_evaluations_html(
    evaluations: Dict[str, EvaluationReport],
    question: str,
    answer: str,
) -> str:
    """
    HTML card with the evaluations of one Q&A pair, for embedding in the
    conversation / single-Q&A exports. Returns "" when the pair was never
    evaluated, so callers can pass the result through unconditionally.
    """
    reports = evaluations_for_pair(evaluations, question, answer)
    if not reports:
        return ""
    return f"""
  <article class="card chat-card shadow-sm">
    <div class="card-header bg-white">
      <span class="badge text-bg-warning badge-role">Evaluation</span>
    </div>
    <div class="card-body">
      {''.join(_judge_blocks_html(reports))}
    </div>
  </article>"""


def render_evaluations_html(
    evaluations: Dict[str, EvaluationReport],
    history: Sequence[Tuple[str, str]],
    title: str = "RAG Evaluation Report",
) -> str:
    """
    Full standalone HTML document with one section per question.

    Mirrors the visual style of `build_conversation_html` from
    `utils.html_export` -- same Bootstrap CDN, same card pattern -- so
    it looks like the rest of the app's exports when opened in a browser.
    """
    lookup = _history_lookup(history)
    by_question: Dict[str, Dict[str, EvaluationReport]] = {}
    for key, report in evaluations.items():
        try:
            q_hash, judge_model = split_eval_key(key)
        except ValueError:
            continue
        by_question.setdefault(q_hash, {})[judge_model] = report

    sections: List[str] = []
    for q_hash in sorted(by_question.keys()):
        q, a = lookup.get(q_hash, ("", ""))
        q_html = _render_rich_html(q) if q else f"<em>(question text not in current chat history; hash <code>{q_hash}</code>)</em>"
        a_html = _render_rich_html(a) if a else ""

        judge_blocks = _judge_blocks_html(by_question[q_hash])

        sections.append(f"""
<article class="card chat-card shadow-sm mb-4">
  <div class="card-header bg-white"><span class="badge text-bg-primary badge-role">Question</span></div>
  <div class="card-body"><div class="card-text">{q_html}</div></div>
  {f'<div class="card-header bg-white"><span class="badge text-bg-success badge-role">Answer</span></div><div class="card-body"><div class="card-text">{a_html}</div></div>' if a_html else ''}
  <div class="card-body">
    <h5 class="card-title h6">Evaluations</h5>
    {''.join(judge_blocks) if judge_blocks else '<p class="text-muted">No evaluations recorded for this question.</p>'}
  </div>
</article>""")

    body = (
        _bootstrap_head(title)
        + '<div class="container py-4">'
        + f'<header class="mb-4"><h1 class="h3">{html.escape(title)}</h1>'
        + f'<p class="text-muted mb-0">{len(by_question)} question'
        + ('s' if len(by_question) != 1 else '')
        + ' · '
        + f'{len(evaluations)} evaluation'
        + ('s' if len(evaluations) != 1 else '')
        + ' cached</p></header>'
        + "".join(sections)
        + "</div>"
        + _bootstrap_foot()
    )
    return body
