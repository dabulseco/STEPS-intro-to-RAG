"""
Unit tests for the LLM-as-judge evaluator.

Nothing in this file needs Ollama. All judge calls are monkeypatched at the
`requests.post` layer, which keeps the suite fast and deterministic.

Run with:  pytest tests/test_evaluator.py -q
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core import eval as ev
from rag_core.evaluator_prompts import (
    CONTEXT_PRECISION_SYSTEM,
    CONTEXT_RECALL_SYSTEM,
    FAITHFULNESS_SYSTEM,
    GROUNDEDNESS_SYSTEM,
    METRICS,
    RELEVANCE_SYSTEM,
    RESPONSE_FORMAT,
    render_user_message,
)


# -----------------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_question() -> str:
    return "What are the key features of Salesforce CRM Sales Cloud?"


@pytest.fixture
def sample_context() -> str:
    return "Salesforce Sales Cloud includes lead management, opportunity tracking, and forecasting."


@pytest.fixture
def sample_answer() -> str:
    return "Sales Cloud offers lead management, opportunity tracking, and AI-driven forecasting."


def _valid_payload(rating: int = 4) -> str:
    """A well-formed judge response -- what a model should emit if it obeys the prompt."""
    return json.dumps({
        "steps": "Identified the question and context, then compared the answer against context claims.",
        "explanation": "The answer reflects the context faithfully with minor omissions.",
        "evaluation": "Followed to a good extent.",
        "rating": rating,
    })


def _ok_response(payload: str) -> MagicMock:
    """A mock `requests.post` return value that looks like an Ollama /api/chat reply."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"message": {"content": payload}})
    return resp


# -----------------------------------------------------------------------------
# TestJsonParsing -- the failure mode that hits first on local models
# -----------------------------------------------------------------------------

class TestJsonParsing:
    """The three-stage fallback in parse_metric_payload."""

    def test_direct_json_loads(self):
        result = ev.parse_metric_payload(_valid_payload())
        assert result["rating"] == 4
        assert "steps" in result

    def test_fenced_json(self):
        raw = "Here is my evaluation:\n```json\n" + _valid_payload(5) + "\n```\nDone."
        result = ev.parse_metric_payload(raw)
        assert result["rating"] == 5

    def test_fenced_no_language_tag(self):
        raw = "```\n" + _valid_payload(3) + "\n```"
        result = ev.parse_metric_payload(raw)
        assert result["rating"] == 3

    def test_prose_prefix_then_json(self):
        # No fence; the regex fallback should grab the JSON object.
        raw = "Let me think...\n\n" + _valid_payload(2) + "\n\nThat's my answer."
        result = ev.parse_metric_payload(raw)
        assert result["rating"] == 2

    def test_nested_json_object_outer_parses(self):
        # When the model emits an outer wrapper around the rubric object,
        # the outer wrapper is itself valid JSON and parses cleanly (Stage 1).
        # The rubric keys won't be present in the outer wrapper -- that's
        # the model's bug, not ours -- but the parser must not crash on it.
        inner = {"steps": "a", "explanation": "b", "evaluation": "c", "rating": 4}
        raw = json.dumps({"thoughts": "...", "result": inner})
        result = ev.parse_metric_payload(raw)
        # Outer object parsed; the rubric keys live one level deeper.
        assert "result" in result
        assert result["result"]["rating"] == 4

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            ev.parse_metric_payload("")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            ev.parse_metric_payload(None)

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            ev.parse_metric_payload("The model produced no JSON at all, sorry.")

    @pytest.mark.parametrize("bad_rating", ["abc", None, [1, 2], {"x": 1}])
    def test_invalid_rating_types_become_none(self, bad_rating):
        payload = json.dumps({
            "steps": "x", "explanation": "y", "evaluation": "z", "rating": bad_rating,
        })
        result = ev._build_metric_result("groundedness", payload)
        assert result.rating is None
        assert result.parse_error is not None

    @pytest.mark.parametrize("rating,expected", [("4", 4), ("4.0", 4), (4.6, 4), (3.4, 3), (5, 5), (1, 1)])
    def test_rating_coercion_accepts_strings_and_floats(self, rating, expected):
        assert ev._coerce_rating(rating) == expected

    @pytest.mark.parametrize("rating", [0, 6, -1, 100])
    def test_rating_out_of_range_becomes_none(self, rating):
        assert ev._coerce_rating(rating) is None


# -----------------------------------------------------------------------------
# TestPromptShape -- the five system prompts must contain the required rubric keys
# -----------------------------------------------------------------------------

class TestPromptShape:
    """Snapshot-ish assertions that catch accidental drift if someone edits a prompt."""

    REQUIRED_KEYS = ("steps", "explanation", "evaluation", "rating")

    @pytest.mark.parametrize("name", sorted(METRICS.keys()))
    def test_metric_prompt_contains_all_rubric_keys(self, name):
        prompt = METRICS[name]
        for key in self.REQUIRED_KEYS:
            assert key in prompt, f"Metric '{name}' prompt missing rubric key '{key}'"

    @pytest.mark.parametrize("name", sorted(METRICS.keys()))
    def test_metric_prompt_cites_metric_name(self, name):
        # The prompt should mention the metric somewhere so a reader knows
        # what they're looking at when the prompts are reviewed side by side.
        # Each notebook metric's distinguishing phrase is checked below.
        prompt = METRICS[name].lower()
        phrases = {
            "groundedness": "derived only from",
            "relevance": "relevance",
            "faithfulness": "inferred",
            "context_precision": "useful",
            "context_recall": "attributed",
        }
        assert phrases[name] in prompt, (
            f"Metric '{name}' prompt missing expected phrase '{phrases[name]}'"
        )

    @pytest.mark.parametrize("name", sorted(METRICS.keys()))
    def test_metric_prompt_excludes_absence_statements(self, name):
        # "The documents do not list X" must not be scored as an
        # unattributable claim -- every metric carries the scope rule.
        from rag_core.evaluator_prompts import _SCOPE
        assert _SCOPE.strip() in METRICS[name]
        assert "do not contain" in METRICS[name]

    def test_response_format_is_in_every_prompt(self):
        # The shared rubric lives in RESPONSE_FORMAT -- verify it is the
        # same string concatenated into every metric prompt.
        for name, prompt in METRICS.items():
            assert RESPONSE_FORMAT.rstrip() in prompt, (
                f"Metric '{name}' did not include RESPONSE_FORMAT verbatim"
            )

    def test_user_message_template_renders_all_three_sections(self, sample_question, sample_context, sample_answer):
        rendered = render_user_message(sample_question, sample_context, sample_answer)
        assert "###Question" in rendered
        assert "###Context" in rendered
        assert "###Answer" in rendered
        assert sample_question in rendered
        assert sample_context in rendered
        assert sample_answer in rendered


# -----------------------------------------------------------------------------
# TestMetricDispatch -- the public entry point dispatches all five metrics
# -----------------------------------------------------------------------------

class TestMetricDispatch:
    """Mock requests.post and assert the evaluator fans out to all five metrics."""

    def _make_post_recorder(self, payloads: List[str]):
        """
        Build a `requests.post` stand-in that returns `payloads[i]` on the i-th call.

        Each payload becomes a /api/chat response body.
        """
        calls: List[Dict] = []

        def fake_post(url, json=None, timeout=None, **kwargs):
            calls.append({"url": url, "json": json, "timeout": timeout})
            idx = len(calls) - 1
            payload = payloads[idx] if idx < len(payloads) else _valid_payload(3)
            return _ok_response(payload)

        return fake_post, calls

    def test_all_five_metrics_dispatched_in_order(self, sample_question, sample_context, sample_answer):
        # Five distinct ratings so we can confirm dispatch order.
        ratings = [1, 2, 3, 4, 5]
        payloads = [_valid_payload(r) for r in ratings]
        fake_post, calls = self._make_post_recorder(payloads)

        progress_calls: List[tuple] = []
        with patch("rag_core.eval.requests.post", side_effect=fake_post):
            report = ev.evaluate_answer(
                question=sample_question,
                answer=sample_answer,
                context_text=sample_context,
                sources=[],
                judge_model="llama3.1:8b",
                progress_callback=lambda name, done, total: progress_calls.append((name, done, total)),
            )

        # All five metrics returned.
        assert set(report.metrics.keys()) == {
            "groundedness", "relevance", "faithfulness",
            "context_precision", "context_recall",
        }
        # Ratings were preserved in dispatch order.
        assert [report.rating(m) for m in METRICS] == ratings
        # Five HTTP calls, one per metric.
        assert len(calls) == 5
        # All calls hit the chat endpoint.
        assert all(c["url"].endswith("/api/chat") for c in calls)
        # The judge model name was sent in every request.
        assert all(c["json"]["model"] == "llama3.1:8b" for c in calls)
        # System prompt + user prompt were populated for each metric.
        for c in calls:
            assert c["json"]["messages"][0]["role"] == "system"
            assert c["json"]["messages"][1]["role"] == "user"
            assert sample_question in c["json"]["messages"][1]["content"]
            assert sample_context in c["json"]["messages"][1]["content"]
            assert sample_answer in c["json"]["messages"][1]["content"]
        # Progress callback fired once per metric, with monotonically increasing `done`.
        assert [done for _, done, _ in progress_calls] == [1, 2, 3, 4, 5]
        # Elapsed time was recorded.
        assert report.elapsed_seconds >= 0
        assert report.judge_model == "llama3.1:8b"

    def test_subset_of_metrics_dispatched(self, sample_question, sample_context, sample_answer):
        fake_post, calls = self._make_post_recorder([_valid_payload(4), _valid_payload(5)])
        with patch("rag_core.eval.requests.post", side_effect=fake_post):
            report = ev.evaluate_answer(
                question=sample_question,
                answer=sample_answer,
                context_text=sample_context,
                sources=[],
                judge_model="llama3.1:8b",
                metrics=("groundedness", "faithfulness"),
            )
        assert set(report.metrics.keys()) == {"groundedness", "faithfulness"}
        assert len(calls) == 2

    def test_parse_failure_recorded_not_raised(self, sample_question, sample_context, sample_answer):
        # Dispatch order is METRICS key order: groundedness, relevance,
        # faithfulness, context_precision, context_recall. The 4th call
        # (index 3) hits context_precision -- we feed it garbage there.
        payloads = [_valid_payload(4), _valid_payload(4), _valid_payload(4),
                    "this is not JSON at all", _valid_payload(5)]
        fake_post, _ = self._make_post_recorder(payloads)

        with patch("rag_core.eval.requests.post", side_effect=fake_post):
            report = ev.evaluate_answer(
                question=sample_question,
                answer=sample_answer,
                context_text=sample_context,
                sources=[],
                judge_model="llama3.1:8b",
            )

        # First three succeeded; context_precision (4th) failed to parse;
        # context_recall (5th) succeeded.
        assert report.metrics["groundedness"].parsed
        assert report.metrics["relevance"].parsed
        assert report.metrics["faithfulness"].parsed
        assert not report.metrics["context_precision"].parsed
        assert report.metrics["context_precision"].parse_error is not None
        assert report.metrics["context_recall"].parsed
        # The raw output was preserved for the UI to show.
        assert report.metrics["context_precision"].raw == "this is not JSON at all"


# -----------------------------------------------------------------------------
# TestGuardRails -- judge_model is required; unknown metrics raise early
# -----------------------------------------------------------------------------

class TestGuardRails:

    def test_empty_judge_model_raises(self, sample_question, sample_context, sample_answer):
        with pytest.raises(ValueError, match="judge_model is required"):
            ev.evaluate_answer(
                question=sample_question,
                answer=sample_answer,
                context_text=sample_context,
                sources=[],
                judge_model="",
            )

    def test_whitespace_judge_model_raises(self, sample_question, sample_context, sample_answer):
        with pytest.raises(ValueError, match="judge_model is required"):
            ev.evaluate_answer(
                question=sample_question,
                answer=sample_answer,
                context_text=sample_context,
                sources=[],
                judge_model="   ",
            )

    def test_unknown_metric_raises(self, sample_question, sample_context, sample_answer):
        with pytest.raises(ValueError, match="Unknown metric"):
            ev.evaluate_answer(
                question=sample_question,
                answer=sample_answer,
                context_text=sample_context,
                sources=[],
                judge_model="llama3.1:8b",
                metrics=("groundedness", "totally_made_up"),
            )


# -----------------------------------------------------------------------------
# TestCacheKey -- session-state key shape and judge-model distinction
# -----------------------------------------------------------------------------

class TestCacheKey:
    """The cache key must distinguish judge models so the comparison workflow works."""

    def test_same_inputs_same_key(self):
        k1 = ev.make_eval_key("Q?", "A.", "llama3.1:8b")
        k2 = ev.make_eval_key("Q?", "A.", "llama3.1:8b")
        assert k1 == k2

    def test_different_judge_models_yield_different_keys(self):
        k1 = ev.make_eval_key("Q?", "A.", "llama3.1:8b")
        k2 = ev.make_eval_key("Q?", "A.", "qwen2.5:14b")
        assert k1 != k2

    def test_different_answers_yield_different_keys(self):
        k1 = ev.make_eval_key("Q?", "A1.", "llama3.1:8b")
        k2 = ev.make_eval_key("Q?", "A2.", "llama3.1:8b")
        assert k1 != k2

    def test_different_questions_yield_different_keys(self):
        k1 = ev.make_eval_key("Q1?", "A.", "llama3.1:8b")
        k2 = ev.make_eval_key("Q2?", "A.", "llama3.1:8b")
        assert k1 != k2

    def test_key_has_expected_shape(self):
        key = ev.make_eval_key("Q?", "A.", "llama3.1:8b")
        h, judge = ev.split_eval_key(key)
        assert len(h) == 12
        assert judge == "llama3.1:8b"
        # Hash portion is hex.
        int(h, 16)  # raises if not hex

    def test_split_malformed_raises(self):
        with pytest.raises(ValueError):
            ev.split_eval_key("nokeyseparator")


# ---------------------------------------------------------------------------
# Evaluations embedded in the conversation / single-Q&A HTML exports
# ---------------------------------------------------------------------------

class TestEvaluationsInHtmlExports:
    Q1, A1 = "What is CRISPR?", "A gene-editing tool."
    Q2, A2 = "What is PCR?", "A DNA amplification method."

    @staticmethod
    def _report(judge: str, rating: int) -> ev.EvaluationReport:
        return ev.EvaluationReport(
            metrics={
                "groundedness": ev.MetricResult(
                    name="groundedness", rating=rating,
                    evaluation="Well supported", explanation="Every claim <cited>.",
                ),
                "relevance": ev.MetricResult(
                    name="relevance", rating=None, parse_error="no JSON found",
                ),
            },
            judge_model=judge,
            elapsed_seconds=2.5,
        )

    def _evals(self):
        return {
            ev.make_eval_key(self.Q1, self.A1, "llama3"): self._report("llama3", 5),
            ev.make_eval_key(self.Q1, self.A1, "mistral"): self._report("mistral", 2),
        }

    def test_unevaluated_pair_renders_nothing(self):
        from utils.eval_render import render_pair_evaluations_html
        assert render_pair_evaluations_html(self._evals(), self.Q2, self.A2) == ""
        assert render_pair_evaluations_html({}, self.Q1, self.A1) == ""

    def test_pair_fragment_includes_every_judge_and_details(self):
        from utils.eval_render import render_pair_evaluations_html
        out = render_pair_evaluations_html(self._evals(), self.Q1, self.A1)
        assert "llama3" in out and "mistral" in out
        assert "5 / 5" in out and "2 / 5" in out
        assert "Groundedness" in out and "Well supported" in out
        assert "Every claim &lt;cited&gt;." in out   # escaped
        assert "Parse error: no JSON found" in out

    def test_pair_export_includes_evaluation_only_when_evaluated(self):
        from utils.eval_render import render_pair_evaluations_html
        from utils.html_export import build_pair_html
        evals = self._evals()
        with_eval = build_pair_html(
            0, self.Q1, self.A1,
            evaluation_html=render_pair_evaluations_html(evals, self.Q1, self.A1),
        )
        without = build_pair_html(
            1, self.Q2, self.A2,
            evaluation_html=render_pair_evaluations_html(evals, self.Q2, self.A2),
        )
        assert "Evaluation</span>" in with_eval and "llama3" in with_eval
        assert "Evaluation</span>" not in without
        # Default call (no evaluation argument) still works unchanged.
        assert "Evaluation</span>" not in build_pair_html(0, self.Q1, self.A1)

    def test_conversation_export_places_evaluation_under_its_answer(self):
        from utils.eval_render import render_pair_evaluations_html
        from utils.html_export import build_conversation_html
        history = [(self.Q1, self.A1), (self.Q2, self.A2)]
        evals = self._evals()
        out = build_conversation_html(
            history,
            evaluation_html=[render_pair_evaluations_html(evals, q, a) for q, a in history],
        )
        first, second = out.split('id="qa-1"')
        assert "llama3" in first and "Evaluation</span>" in first
        assert "Evaluation</span>" not in second
        assert out.count(">evaluated</span>") == 1
        # Default call (no evaluations) still works unchanged.
        assert "Evaluation</span>" not in build_conversation_html(history)
