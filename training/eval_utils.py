"""
Reward functions for verl GRPO training on MATH.

Two reward functions:
  grpo_reward_standard  -- correctness only (+1 correct, 0 incorrect)
  grpo_reward_wager     -- proper scoring rule reward using declared confidence R:
                           correct: 2R - R^2,  incorrect: -R^2
                           R is parsed from "Confidence: X" as the first line
                           of the model's <think> block.

Both functions follow the verl reward interface:
    fn(data_source, solution_str, ground_truth, extra_info) -> dict
where "score" is the training reward and all other keys are logged to wandb.
"""

import re

from math_verify import parse, verify


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> str | None:
    """Extract content of the last \\boxed{} in text, handling nested braces."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def _check_correct(pred: str, gold: str) -> bool:
    """Check mathematical equivalence via math_verify, falling back to string match."""
    try:
        return verify(parse(pred, parsing_timeout=None), parse(gold, parsing_timeout=None))
    except Exception:
        if pred is None or gold is None:
            return False
        return pred.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Confidence extraction (wager only)
# ---------------------------------------------------------------------------

_CONFIDENCE_RE = re.compile(
    r"^\s*Confidence:\s*([0-9]{1,3})", re.IGNORECASE | re.MULTILINE
)


def _word_prefix(text: str, max_words: int) -> str:
    """Return the character prefix of text that spans at most max_words whitespace-delimited words."""
    count = 0
    i = 0
    n = len(text)
    while i < n and count < max_words:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        while i < n and not text[i].isspace():
            i += 1
        count += 1
    return text[:i]


def _extract_confidence(response: str, max_words: int | None = None) -> float | None:
    """
    Parse the declared confidence from the response. Rescales integer 0-100 to [0, 1].
    If max_words is set, only searches within the first max_words whitespace-delimited words.
    Returns None if not found or value is out of range.
    """
    search_text = _word_prefix(response, max_words) if max_words is not None else response
    m = _CONFIDENCE_RE.search(search_text)
    if m is None:
        return None
    try:
        x = int(m.group(1))
    except ValueError:
        return None
    if x < 0 or x > 100:
        return None
    return x / 100.0


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def grpo_reward_standard(data_source, solution_str, ground_truth, extra_info):
    """
    Standard correctness reward: +1 if correct, 0 otherwise.

    Args:
        data_source:  str, dataset identifier (unused here)
        solution_str: str, full model response
        ground_truth: str, expected answer
        extra_info:   dict, metadata (subject, level)

    Returns:
        dict with "score" and per-rollout metrics for wandb logging.
    """
    pred = _extract_boxed(solution_str)
    is_correct = int(_check_correct(pred, ground_truth)) if pred is not None else 0

    return {
        "score": float(is_correct),
        "is_correct": is_correct,
        "response_length": len(solution_str),
    }


def grpo_reward_wager(data_source, solution_str, ground_truth, extra_info):
    """
    Proper scoring rule reward using the model's declared confidence R.

    The model is expected to begin its <think> block with:
        Confidence: X
    where X is an integer in [0, 100], rescaled to [0, 1] before reward computation.

    Reward:
        correct:   2R - R^2        (maximised at R = p, expected reward = p^2)
        incorrect: -R^2            (penalises overconfidence)

    If no confidence is declared, score = -1.0.

    Logged metrics (averaged by verl across rollouts):
        declared_confidence  -- R declared by the model
        is_correct           -- 1 if answer correct, 0 otherwise
        confidence_declared  -- 1 if model declared a confidence, 0 if missing
        response_length      -- length of full response string
    """
    r = _extract_confidence(solution_str)
    confidence_declared = int(r is not None)

    pred = _extract_boxed(solution_str)
    is_correct = int(_check_correct(pred, ground_truth)) if pred is not None else 0

    if r is None:
        score = -1.0
    elif is_correct:
        score = 2 * r - r ** 2
    else:
        score = -(r ** 2)

    return {
        "score": score,
        "declared_confidence": r,
        "is_correct": is_correct,
        "confidence_declared": confidence_declared,
        "response_length": len(solution_str),
    }


def grpo_reward_wager_cot(data_source, solution_str, ground_truth, extra_info):
    """
    Proper scoring rule reward for the CoT-sketch variant.

    Same as grpo_reward_wager but the model is allowed up to 100 words of
    planning before declaring confidence. Confidence must appear within the
    first 100 words; anything later is treated as undeclared (score = -1.0).
    """
    r = _extract_confidence(solution_str, max_words=100)
    confidence_declared = int(r is not None)

    pred = _extract_boxed(solution_str)
    is_correct = int(_check_correct(pred, ground_truth)) if pred is not None else 0

    if r is None:
        score = -1.0
    elif is_correct:
        score = 2 * r - r ** 2
    else:
        score = -(r ** 2)

    return {
        "score": score,
        "declared_confidence": r,
        "is_correct": is_correct,
        "confidence_declared": confidence_declared,
        "response_length": len(solution_str),
    }
