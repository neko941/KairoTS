"""Answer parsing and scoring of FinTSR-Bench generations.

official code (`src/evaluation/utils.py`, `get_score.py`; the paper defines success
rate as the share of outputs with a valid answer tag): the text between the first
`<answer>` and `</answer>` tags, stripped of everything but letters and spaces, is
compared case-insensitively with the gold letter. An output without a parsable tag is
invalid and scores 0, so accuracy counts it wrong.
"""

import re
from collections.abc import Iterable

from kairots.reasoning.FinTSRBench import TASKS


def extract_answer(*, text: str) -> str | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match is None:
        return None
    letters = re.sub(r"[^a-zA-Z\s]", "", match.group(1).strip())
    return letters or None


def reward(*, text: str | None, answer: str) -> tuple[float, bool]:
    """(1.0 or 0.0, valid)."""
    if not text or text == "null":
        return 0.0, False
    parsed = extract_answer(text=text)
    if parsed is None:
        return 0.0, False
    return float(parsed.strip().upper() == answer.strip().upper()), True


def score(*, results: Iterable[tuple[str, str, str | None]]) -> dict[str, float]:
    """Metrics from (task, gold answer, generated text) triples: per-task accuracy
    and success rate (%), the task mean (the paper's Avg. column) and the sample
    mean (the official `overall_score`)."""
    rewards: dict[str, list[float]] = {}
    valid: dict[str, list[bool]] = {}
    for task, answer, text in results:
        value, is_valid = reward(text=text, answer=answer)
        rewards.setdefault(task, []).append(value)
        valid.setdefault(task, []).append(is_valid)
    metrics: dict[str, float] = {}
    for task in TASKS:
        if task.key in rewards:
            values = rewards[task.key]
            metrics[f"accuracy/{task.key}"] = 100.0 * sum(values) / len(values)
            metrics[f"success_rate/{task.key}"] = (
                100.0 * sum(valid[task.key]) / len(values)
            )
    seen = [task.key for task in TASKS if task.key in rewards]
    if seen:
        metrics["accuracy_task_mean"] = sum(
            metrics[f"accuracy/{key}"] for key in seen
        ) / len(seen)
        every = [value for key in seen for value in rewards[key]]
        metrics["accuracy"] = 100.0 * sum(every) / len(every)
        flags = [flag for key in seen for flag in valid[key]]
        metrics["success_rate"] = 100.0 * sum(flags) / len(flags)
    return metrics
