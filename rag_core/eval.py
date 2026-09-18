"""
LLM-as-judge evaluation of a single RAG answer.

Given a question, the assembled context, the model's answer, and a chosen
Ollama chat model, run the five notebook metrics (groundedness, relevance,
faithfulness, context precision, context recall) and return an
`EvaluationReport`.

Design constraints (mirroring the rest of `rag_core`):

* No Streamlit, no LangChain -- the judge calls `requests.post` against the
  Ollama HTTP API directly, so this module is testable without a browser or
  any of the app's UI machinery.
* `judge_model` is required. There is intentionally no default: the app's
  sidebar forces the user to pick one, and silent fallbacks would mask the
  self-evaluation bias the user is supposed to be able to avoid.
* Three-stage JSON-parse fallback -- local models routinely wrap output in
  ```json fences or prefix it with prose, and a strict `json.loads` would
  silently zero out every rating. The fallback chain is documented at
  `parse_metric_payload`.
* Sequential dispatch. Ollama's `/api/generate` serializes VRAM, and the
  cost of fan-out across local threads is a higher OOM risk than the wall-
  clock saving on consumer GPUs.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import requests

from .config import OLLAMA_BASE_URL
from .context_assembler import Source
from .evaluator_prompts import METRICS, render_user_message


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """One metric's verdict on a single Q&A pair."""

    name: str
    rating: Optional[int]            # 1-5, or None on parse failure
    steps: str = ""
    explanation: str = ""
    evaluation: str = ""              # short summary phrase shown in the UI
    raw: str = ""                    # original judge output, for debugging
    parse_error: Optional[str] = None

    @property
    def parsed(self) -> bool:
        """True iff the rating was extracted successfully."""
        return self.rating is not None


@dataclass
class EvaluationReport:
    """All five metric verdicts for one (question, answer, judge_model) triple."""

    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    judge_model: str = ""
    elapsed_seconds: float = 0.0

    def rating(self, metric: str) -> Optional[int]:
        """Convenience accessor -- `report.rating("groundedness")`."""
        m = self.metrics.get(metric)
        return m.rating if m else None


# ---------------------------------------------------------------------------
# JSON parsing -- the failure mode that hits first on local models
# ---------------------------------------------------------------------------

# Matches a ```json ... ``` or ``` ... ``` fence anywhere in the output.
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

# Matches the first balanced-looking {...} block when no fence is present.
# Greedy enough to span nested objects but stops at the first closing brace
# at column 0, which is what well-behaved models emit.
_FIRST_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def parse_metric_payload(raw: str) -> Dict[str, object]:
    """
    Extract the rubric JSON from a judge's raw output.

    Three-stage fallback:

    1. Direct `json.loads(raw)` -- works when the model obeyed the prompt.
    2. Strip ```json / ``` fences, retry on the fenced body.
    3. Grab the first balanced `{...}` block via regex, retry.

    Raises:
        ValueError: if all three stages fail. The caller is expected to
            catch and surface the raw output to the user rather than
            silently treating the rating as zero.
    """
    if raw is None:
        raise ValueError("Judge returned empty output (None).")

    text = raw.strip()
    if not text:
        raise ValueError("Judge returned empty output.")

    # Stage 1: well-behaved output.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Stage 2: fenced output. We try every fence in case there is more than
    # one (rare, but cheap to handle).
    for match in _FENCE_RE.finditer(text):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

    # Stage 3: regex-extract the first {...} block.
    blob = _FIRST_OBJECT_RE.search(text)
    if blob:
        try:
            return json.loads(blob.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract JSON from judge output (first 200 chars): {text[:200]!r}"
    )


def _coerce_rating(value: object) -> Optional[int]:
    """
    Turn whatever the model emitted into an int 1-5, or None if it isn't one.

    Models occasionally emit floats ("4.0"), strings ("4"), or out-of-range
    integers. The function normalizes without raising so the rest of the
    pipeline can carry on.
    """
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            return None
    if 1 <= n <= 5:
        return n
    return None


def _build_metric_result(name: str, raw: str) -> MetricResult:
    """Parse one judge's raw output into a MetricResult, recording any failure."""
    try:
        payload = parse_metric_payload(raw)
    except ValueError as exc:
        return MetricResult(
            name=name,
            rating=None,
            raw=raw,
            parse_error=str(exc),
        )

    rating = _coerce_rating(payload.get("rating"))
    return MetricResult(
        name=name,
        rating=rating,
        steps=str(payload.get("steps", "") or ""),
        explanation=str(payload.get("explanation", "") or ""),
        evaluation=str(payload.get("evaluation", "") or ""),
        raw=raw,
        parse_error=None if rating is not None else "Rating missing or out of range.",
    )


# ---------------------------------------------------------------------------
# Ollama HTTP call
# ---------------------------------------------------------------------------

class JudgeCallError(RuntimeError):
    """Raised when the judge model call to Ollama fails."""


def _call_judge(
    judge_model: str,
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    temperature: float,
    timeout: int,
) -> str:
    """
    Invoke the Ollama chat-completions endpoint and return the assistant text.

    Using `/api/chat` (not `/api/generate`) because the notebook's prompts
    are framed as system+user turns and `/api/chat` is the natural fit.
    """
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise JudgeCallError(
            f"Judge call to '{judge_model}' failed: {exc}"
        ) from exc

    data = response.json()
    message = data.get("message") or {}
    content = message.get("content")
    if not content:
        raise JudgeCallError(
            f"Judge call to '{judge_model}' returned no message content."
        )
    return content


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[str, int, int], None]


def evaluate_answer(
    question: str,
    answer: str,
    context_text: str,
    sources: Sequence[Source],
    judge_model: str,
    base_url: str = OLLAMA_BASE_URL,
    metrics: Sequence[str] = (
        "groundedness",
        "relevance",
        "faithfulness",
        "context_precision",
        "context_recall",
    ),
    temperature: float = 0.0,
    progress_callback: Optional[ProgressCallback] = None,
    timeout: int = 180,
) -> EvaluationReport:
    """
    Score a single Q&A pair across all five metrics.

    Args:
        question: The user's question.
        answer: The model's answer text.
        context_text: The retrieved-context block fed to the answer model.
            (Kept as a string rather than re-derived from `sources` so the
            caller controls exactly what the judge sees.)
        sources: The numbered sources, in order. Currently unused by the
            five notebook metrics but accepted so a future metric can
            inspect `[S1]`-style markers without an API change.
        judge_model: REQUIRED. Name of the Ollama chat model that will
            act as judge. Must be non-empty.
        base_url: Ollama server URL.
        metrics: Subset of `METRICS` keys to run, in dispatch order.
        temperature: Sampling temperature for the judge. Defaults to 0
            (matches the notebook).
        progress_callback: Called as `(metric_name, completed, total)`
            after each metric completes, so the UI can drive a progress
            bar without this module importing Streamlit.
        timeout: Per-call HTTP timeout in seconds.

    Returns:
        EvaluationReport with one MetricResult per requested metric.

    Raises:
        ValueError: if `judge_model` is empty, or if any metric name in
            `metrics` is not in `METRICS`.
        JudgeCallError: on Ollama HTTP failures (caller decides whether
            to surface as a hard error or to record a stub MetricResult).
    """
    if not judge_model or not judge_model.strip():
        raise ValueError("judge_model is required -- the evaluator does not pick a default.")

    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {unknown}. Available: {sorted(METRICS.keys())}"
        )

    user_prompt = render_user_message(
        question=question,
        context=context_text,
        answer=answer,
    )

    report = EvaluationReport(judge_model=judge_model)
    started = time.monotonic()
    total = len(metrics)

    for index, name in enumerate(metrics, start=1):
        system_prompt = METRICS[name]
        raw = _call_judge(
            judge_model=judge_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
        )
        report.metrics[name] = _build_metric_result(name, raw)

        if progress_callback is not None:
            progress_callback(name, index, total)

    report.elapsed_seconds = time.monotonic() - started
    return report


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------
#
# The session-state cache for evaluation reports is keyed on
# (question, answer, judge_model) so that:
#   - re-running with the same model is a cache hit (free re-render);
#   - re-running with a different judge model produces a distinct entry,
#     enabling the "compare judge models on the same Q&A" workflow.
#
# A single helper here means there is exactly one place where the key shape
# is defined -- Phase 2 (Streamlit UI) imports this rather than reinventing it.
# ---------------------------------------------------------------------------

_HASH_SEP = "\x00"   # cannot appear in user-typed text


def make_eval_key(question: str, answer: str, judge_model: str) -> str:
    """
    Build a stable cache key for one (question, answer, judge_model) triple.

    Returns "<12-char-hash>__<judge_model>". The hash uses SHA-1 truncated
    to 12 hex chars; collisions are vanishingly unlikely for chat-history
    scale, and a collision only causes a harmless re-evaluation.
    """
    h = hashlib.sha1(f"{question}{_HASH_SEP}{answer}".encode("utf-8")).hexdigest()[:12]
    return f"{h}__{judge_model}"


def split_eval_key(key: str) -> tuple[str, str]:
    """
    Inverse of `make_eval_key` -- returns (question_hash, judge_model).

    Raises ValueError if the key was not produced by `make_eval_key`. Callers
    that pass through user-supplied keys should guard this.
    """
    if "__" not in key:
        raise ValueError(f"Malformed eval key (no separator): {key!r}")
    h, judge = key.split("__", 1)
    return h, judge
