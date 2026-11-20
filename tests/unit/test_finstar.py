import json
from pathlib import Path
from typing import Any

import pytest
import torch

from kairots.reasoning import FinSTaR, FinTSRBench, scoring

PROMPT = "Stock X closes: [1.0, 2.0, 3.0]. Classify the trend. (A) Up (B) Down"


@pytest.fixture(autouse=True)
def _no_broken_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """transformers asks `torch.cuda.is_current_stream_capturing()` on every
    forward; on a machine whose CUDA runtime is broken (cudaErrorNotSupported) that
    call crashes the process, so CPU-only tests answer it themselves."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def _item(*, task: str, answer: str, cot: bool = True) -> dict[str, Any]:
    target = (
        f"<think>\nStep 1: compute.\n</think>\n<answer>({answer})</answer>"
        if cot
        else f"<answer>({answer})</answer>"
    )
    return {
        "task": task,
        "answer": answer,
        "conversations": [
            {"role": "user", "content": f"{PROMPT} [{task}]"},
            {"role": "assistant", "content": target},
        ],
        "metadata": {},
    }


def test_answer_parsing_follows_the_official_scorer() -> None:
    assert (
        scoring.extract_answer(text="<think>x</think>\n<answer> (C) </answer>") == "C"
    )
    assert scoring.extract_answer(text="no tag (C)") is None
    assert scoring.extract_answer(text="<answer>(1)</answer>") is None
    assert scoring.reward(text="<answer>(b)</answer>", answer="B") == (1.0, True)
    assert scoring.reward(text="<answer>(A)</answer>", answer="B") == (0.0, True)
    assert scoring.reward(text="", answer="B") == (0.0, False)
    assert scoring.reward(text="(B)", answer="B") == (0.0, False)


def test_scores_count_invalid_outputs_wrong_and_average_tasks() -> None:
    metrics = scoring.score(
        results=[
            ("F1_drawdown", "A", "<answer>(A)</answer>"),
            ("F1_drawdown", "B", "garbage"),
            ("F3_event", "B", "<answer>(B)</answer>"),
        ]
    )
    assert metrics["accuracy/F1_drawdown"] == 50.0
    assert metrics["success_rate/F1_drawdown"] == 50.0
    assert metrics["accuracy/F3_event"] == 100.0
    # Paper Avg.: the mean of task accuracies; official overall: the sample mean.
    assert metrics["accuracy_task_mean"] == 75.0
    assert metrics["accuracy"] == pytest.approx(200.0 / 3)
    assert metrics["success_rate"] == pytest.approx(200.0 / 3)


def test_loading_keeps_the_last_turns_and_answer_only_strips_the_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train_cot.json"
    path.write_text(json.dumps([_item(task="F6_trend", answer="E")]), encoding="utf-8")
    (sample,) = FinTSRBench.load(path=path)
    assert sample.prompt.startswith("Stock X") and sample.target is not None
    assert "<think>" in sample.target
    (bare,) = FinTSRBench.load(path=path, answers_only=True)
    assert bare.target == "<answer>(E)</answer>"
    path.write_text(json.dumps([_item(task="F99", answer="A")]), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        FinTSRBench.load(path=path)


def test_class_weights_are_the_official_inverse_frequencies() -> None:
    samples = [
        FinTSRBench.Sample(task=task, answer=answer, prompt="", target="")
        for task, answer in [("F3_event", "A")] * 3
        + [("F3_event", "B")]
        + [("F1_drawdown", "C")]
    ]
    weights = FinTSRBench.class_weights(samples=samples)
    assert weights == [4 / 2 / 3] * 3 + [4 / 2 / 1] + [1.0]


def test_weighted_loss_is_the_weighted_mean_of_per_sample_token_means() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, -100]])
    weights = torch.tensor([2.0, 0.5])
    shift = logits[:, :-1]
    target = labels[:, 1:]
    first = torch.nn.functional.cross_entropy(shift[0], target[0])
    second = torch.nn.functional.cross_entropy(shift[1, 1:2], target[1, 1:2])
    expected = (2.0 * first + 0.5 * second) / 2
    loss = FinSTaR.sample_loss(logits=logits, labels=labels, weights=weights)
    torch.testing.assert_close(loss, expected)
    unweighted = FinSTaR.sample_loss(logits=logits, labels=labels, weights=None)
    torch.testing.assert_close(
        unweighted,
        torch.nn.functional.cross_entropy(
            shift.reshape(-1, 5), target.reshape(-1), ignore_index=-100
        ),
    )


def test_paper_scores_the_final_adapter_code_every_epoch() -> None:
    assert FinSTaR.Config().evaluated_epochs() == [4]
    assert FinSTaR.Config(reference="code").evaluated_epochs() == [1, 2, 3, 4]
    assert FinSTaR.Config(train="").evaluated_epochs() == [0]
    with pytest.raises(ValueError, match="splits"):
        FinSTaR.Config(splits=("D",))


def _tiny_backbone(*, directory: Path, texts: list[str]) -> None:
    """A random two-layer Qwen2 and a character tokenizer with the ChatML template,
    built offline."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    specials = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"]
    characters = sorted(
        {character for text in texts for character in text}
        | set("userassistant\n()ABCDE<>/answerthink")
    )
    vocabulary = {token: index for index, token in enumerate(specials + characters)}
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocabulary, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(r"[\s\S]"), behavior="isolated"
    )
    tokenizer.add_special_tokens(specials)
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
    )
    fast.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
        "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}"
        "<|im_start|>assistant\n{% endif %}"
    )
    fast.save_pretrained(directory)
    config = transformers.Qwen2Config(
        vocab_size=len(vocabulary),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        # A frozen head at the default 0.02 scale bounds every logit near +-0.6,
        # so LoRA (which leaves the head and norms frozen) could not fit a target.
        initializer_range=1.0,
        eos_token_id=vocabulary["<|im_end|>"],
        pad_token_id=vocabulary["<pad>"],
    )
    torch.manual_seed(0)
    transformers.Qwen2ForCausalLM(config).save_pretrained(directory)


def test_encode_masks_the_prompt_and_trains_on_the_answer(tmp_path: Path) -> None:
    _tiny_backbone(directory=tmp_path / "model", texts=[PROMPT])
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "model")
    item = FinSTaR.encode(
        tokenizer=tokenizer,
        prompt=PROMPT,
        target="<answer>(A)</answer>",
        max_length=512,
    )
    kept = item["labels"][item["labels"] != -100]
    # The target and its end-of-turn token are trained; the prompt is not.
    assert tokenizer.decode(kept).rstrip("\n") == "<answer>(A)</answer><|im_end|>"
    short = FinSTaR.encode(
        tokenizer=tokenizer, prompt=PROMPT, target="<answer>(A)</answer>", max_length=10
    )
    # official code: a marker cut off by truncation leaves nothing masked.
    assert len(short["input_ids"]) == 10
    torch.testing.assert_close(short["labels"], short["input_ids"])


def test_tiny_model_pipeline_trains_generates_and_scores(tmp_path: Path) -> None:
    items = [
        _item(task=task, answer=answer)
        for task, answer in [("F6_trend", "A"), ("F3_event", "B")] * 2
    ]
    data = tmp_path / "FinTSR-Bench"
    data.mkdir()
    for name in ("train_cot.json", "test_sft.json"):
        (data / name).write_text(json.dumps(items), encoding="utf-8")
    texts = [message["content"] for item in items for message in item["conversations"]]
    _tiny_backbone(directory=tmp_path / "model", texts=texts)
    config = FinSTaR.Config(
        reference="code",
        model=str(tmp_path / "model"),
        data_dir=str(data),
        splits=("A",),
        lora_rank=4,
        lora_alpha=8,
        learning_rate=1e-2,
        epochs=2,
        gradient_accumulation=2,
        max_length=256,
        precision="fp32",
        max_new_tokens=8,
        device="cpu",
        runs_dir=str(tmp_path / "runs"),
        trust_remote_code=False,
    )
    result = FinSTaR.run(config=config)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "lora" / "epoch_1" / "adapter_config.json").is_file()
    assert (run_dir / "lora" / "epoch_2" / "adapter_config.json").is_file()
    assert set(result["metrics"]) == {"epoch_1/A", "epoch_2/A"}
    for metrics in result["metrics"].values():
        assert {"accuracy", "accuracy_task_mean", "success_rate"} <= set(metrics)
        assert 0.0 <= metrics["accuracy"] <= 100.0
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 * len(items)


def test_lora_learns_the_answer_tag_on_a_repeated_sample(tmp_path: Path) -> None:
    item = _item(task="F3_event", answer="B", cot=False)
    data = tmp_path / "FinTSR-Bench"
    data.mkdir()
    (data / "train_ao.json").write_text(json.dumps([item] * 8), encoding="utf-8")
    (data / "test_sft.json").write_text(json.dumps([item]), encoding="utf-8")
    texts = [message["content"] for message in item["conversations"]]
    _tiny_backbone(directory=tmp_path / "model", texts=texts)
    config = FinSTaR.Config(
        model=str(tmp_path / "model"),
        data_dir=str(data),
        train="train_ao",
        splits=("A",),
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.0,
        learning_rate=3e-2,
        warmup_fraction=0.0,
        epochs=6,
        gradient_accumulation=1,
        max_length=256,
        precision="fp32",
        gradient_checkpointing=False,
        max_new_tokens=20,
        device="cpu",
        runs_dir=str(tmp_path / "runs"),
        trust_remote_code=False,
    )
    result = FinSTaR.run(config=config)
    assert result["metrics"]["epoch_6/A"]["accuracy"] == 100.0


def test_reasoning_path_is_centralized_and_bills_nothing() -> None:
    """FinSTaR trains one model on the whole training file: no clients, no
    strategy, no Messages, so no traffic is billed."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(FinSTaR))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(
        name.startswith(("kairots.runtime", "kairots.strategies")) for name in imported
    )
    assert not {"strategy", "clients", "iterations"} & {
        item.name for item in __import__("dataclasses").fields(FinSTaR.Config)
    }
