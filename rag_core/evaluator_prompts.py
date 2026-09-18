"""
Five LLM-as-judge system prompts for RAG self-evaluation.

Ported from the JHU MLS Week 8 notebook (Metric 1-5), with two hardenings:
the rubric's JSON schema is declared as `RESPONSE_FORMAT` here so the body of
each prompt and the expected schema cannot drift apart, and `_SCOPE` tells the
judge to ignore statements that the context lacks the requested information.

The user-message template (`###Question / ###Context / ###Answer`) is the
notebook's `user_message_template`, kept identical so behavior matches the
classroom reference exactly.
"""

# ---------------------------------------------------------------------------
# Shared response-format rubric -- referenced by every metric prompt so the
# expected JSON shape is in one place.
# ---------------------------------------------------------------------------
RESPONSE_FORMAT = """\
Output Format:
Arrange your output in the following JSON format.
{
    "steps": write down the steps that are needed to evaluate the answer as per the metric.
    "explanation": provide a step-by-step explanation if the context adheres to the metric considering the question as the input.
    "evaluation": the extent to which the metric is followed.
    "rating": <1,2,3,4,5>
}
DO NOT output anything else before or after the JSON output.
"""

# Shared preface -- every metric prompt begins with this role description and
# evaluation criteria block. The notebook repeats the same preface five times;
# extracting it makes the prompts easier to scan and edit.
_PREFACE = """\
You are tasked with rating AI generated answers to questions posed by users.
You will be presented a question, context used by the AI system to generate the answer and an AI generated answer to the question.
In the input, the question will begin with ###Question, the context will begin with ###Context while the AI generated answer will begin with ###Answer.

Evaluation criteria:
The task is to judge the extent to which the metric is followed by the answer.
1 - The metric is not followed at all
2 - The metric is followed only to a limited extent
3 - The metric is followed to a good extent
4 - The metric is followed mostly
5 - The metric is followed completely

Metric:
"""

# Not in the notebook. RAG answers routinely say "the documents do not list X"
# when retrieval comes up short; judges otherwise mark that sentence as
# "not attributable to the context" (the context never states its own gaps)
# and drag the score down for an honest, correct abstention.
_SCOPE = """\

Scope:
Ignore any statement in the answer that only says the context or documents do not contain, list, mention or provide certain information (for example "The provided documents do not list specific metrics").
Such statements are not claims to be verified: do not evaluate them, do not mention them in your explanation, and do not let them raise or lower the rating.
Rate the answer on its remaining content only. If the answer consists only of such statements, rate it on whether the context indeed lacks the information the question asks for.
"""

_INSTRUCTIONS = """\

Instructions:
1. First write down the steps that are needed to evaluate the answer as per the metric.
2. Give a step-by-step explanation if the answer adheres to the metric considering the question and context as the input.
3. Next, evaluate the extent to which the metric is followed.
4. Use the previous information to rate the answer using the evaluaton criteria and assign a score.
"""


def _build(metric_definition: str, *, instructions_lead: str = "answer") -> str:
    """
    Compose one of the five judge system prompts.

    Args:
        metric_definition: The metric statement, placed after `Metric:` in the preface.
        instructions_lead: Word used in step 2 ("answer" for groundedness/faithfulness,
            "context" for relevance/context_precision/context_recall) -- matches the
            notebook's original phrasing for each metric.
    """
    instructions = _INSTRUCTIONS.replace(
        "explanation if the answer adheres", f"explanation if the {instructions_lead} adheres"
    )
    return _PREFACE + metric_definition + "\n" + _SCOPE + instructions + "\n" + RESPONSE_FORMAT


# ---------------------------------------------------------------------------
# The five metric prompts. Body text is the notebook's Metric 1-5 content.
# Small wording tweaks (in comments) flag where we hardened the rubric.
# ---------------------------------------------------------------------------

GROUNDEDNESS_SYSTEM = _build(
    "The answer should be derived only from the information presented in the context.",
    instructions_lead="answer",
)

RELEVANCE_SYSTEM = _build(
    # Notebook Metric 2 wording, preserved.
    "Relevance measures how well the answer addresses the main aspects of the question, "
    "based on the context.\n"
    "Consider whether all and only the important aspects are contained in the answer when "
    "evaluating relevance.",
    instructions_lead="context",
)

FAITHFULNESS_SYSTEM = _build(
    "The answer can be directly inferred based on the context.",
    instructions_lead="answer",
)

CONTEXT_PRECISION_SYSTEM = _build(
    "Verify if the context was useful in arriving at the given answer.",
    instructions_lead="context",
)

CONTEXT_RECALL_SYSTEM = _build(
    "Analyze each sentence in the answer and classify if the sentence can be attributed "
    "to the given context or not.",
    instructions_lead="context",
)


# Map of metric name -> system prompt. The order in this dict is the order the
# evaluator dispatches the metrics in -- first-to-last matches the notebook's
# Metric 1..5 numbering, so anyone familiar with the notebook gets the same
# order in the UI.
METRICS: "dict[str, str]" = {
    "groundedness":     GROUNDEDNESS_SYSTEM,
    "relevance":        RELEVANCE_SYSTEM,
    "faithfulness":     FAITHFULNESS_SYSTEM,
    "context_precision": CONTEXT_PRECISION_SYSTEM,
    "context_recall":   CONTEXT_RECALL_SYSTEM,
}


# ---------------------------------------------------------------------------
# User-message template -- verbatim from the notebook's `user_message_template`.
# ---------------------------------------------------------------------------
USER_MESSAGE_TEMPLATE = """\
###Question
{question}

###Context
{context}

###Answer
{answer}
"""


def render_user_message(question: str, context: str, answer: str) -> str:
    """Render the user-side prompt for any of the five metrics."""
    return USER_MESSAGE_TEMPLATE.format(
        question=question, context=context, answer=answer
    )
