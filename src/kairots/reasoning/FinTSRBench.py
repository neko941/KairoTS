"""FinTSR-Bench (Lee et al., FinSTaR, arXiv:2605.03460v5): ten multiple-choice
reasoning tasks on 120-day S&P 500 closing prices.

Samples follow the released chat format (`seunghanlee/FinTSR-Bench`, README of
seunghan96/FinSTaR@0ef2040e): a `task` id, the gold `answer` letter, and
`conversations` holding the user prompt (prices serialized as text) and, in the
training files, the assistant target (`<think>...</think><answer>(X)</answer>`).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, kw_only=True)
class Task:
    key: str
    label: str
    category: str
    classes: int


# Paper Table 3 order (the main-results columns): assessment, then prediction.
TASKS = (
    Task(key="F1_drawdown", label="Draw.", category="AS", classes=4),
    Task(key="F5_volatility", label="Vol.", category="AS", classes=3),
    Task(key="F6_trend", label="Trend", category="AS", classes=5),
    Task(key="F2_correlation", label="Corr.", category="AM", classes=3),
    Task(key="F3_event", label="Event", category="PS", classes=2),
    Task(key="F7_breakout", label="S/R", category="PS", classes=2),
    Task(key="F8_drawdown_recovery", label="DDR", category="PS", classes=2),
    Task(key="F9_volatility_forecast", label="V.F.", category="PS", classes=2),
    Task(key="F4_momentum", label="R.P.", category="PM", classes=2),
    Task(key="F10_pair_convergence", label="P.C.", category="PM", classes=2),
)
BY_KEY = {task.key: task for task in TASKS}
# The released files per split (data/README.md of the official repository).
FILES = {
    "train_cot": "train_cot.json",
    "train_ao": "train_ao.json",
    "A": "test_sft.json",
    "B": "test_b_ood_stock.json",
    "C": "test_c_ood_stock_period.json",
}


@dataclass(frozen=True, kw_only=True)
class Sample:
    task: str
    answer: str
    prompt: str
    target: str | None


def _message(*, conversations: list[dict[str, str]], role: str) -> str | None:
    # official code (`FinTSRDatasetV2`): the last message of each role.
    content = None
    for message in conversations:
        if message["role"] == role:
            content = message["content"]
    return content


def answer_only(*, target: str) -> str:
    """official code (`--answer_only`): keep only the `<answer>` tag."""
    match = re.search(r"<answer>(.*?)</answer>", target, re.DOTALL)
    return f"<answer>{match.group(1).strip()}</answer>" if match else target


def load(*, path: str | Path, answers_only: bool = False) -> list[Sample]:
    items: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = []
    for item in items:
        if item["task"] not in BY_KEY:
            raise ValueError(f"{path}: unknown FinTSR-Bench task {item['task']!r}")
        prompt = _message(conversations=item["conversations"], role="user")
        if prompt is None:
            raise ValueError(f"{path}: a sample has no user message")
        target = _message(conversations=item["conversations"], role="assistant")
        if target is not None and answers_only:
            target = answer_only(target=target)
        samples.append(
            Sample(
                task=item["task"], answer=item["answer"], prompt=prompt, target=target
            )
        )
    return samples


def class_weights(*, samples: list[Sample]) -> list[float]:
    """official code (`--class_weighted`, the paper is silent): each sample weighs
    (task count / task classes) / (its (task, answer) count), so every task's
    classes weigh equally on average."""
    counts: dict[tuple[str, str], int] = {}
    for sample in samples:
        key = (sample.task, sample.answer)
        counts[key] = counts.get(key, 0) + 1
    totals: dict[str, int] = {}
    classes: dict[str, int] = {}
    for (task, _), count in counts.items():
        totals[task] = totals.get(task, 0) + count
        classes[task] = classes.get(task, 0) + 1
    return [
        totals[sample.task] / classes[sample.task] / counts[sample.task, sample.answer]
        for sample in samples
    ]
