"""FinSTaR: LoRA supervised fine-tuning of an LLM on FinTSR-Bench chains of thought,
Lee et al. (EMNLP 2026), arXiv:2605.03460v5.

Reference: the paper and seunghan96/FinSTaR@0ef2040e (Apache-2.0). The CoT targets
(Compute-in-CoT for assessment, Scenario-Aware CoT for prediction) come with the
released FinTSR-Bench training file; this module fine-tunes a causal LM on them with
LoRA, generates answers to the test splits and scores them.

The paper is published, so `reference: "paper"` (the default) follows it where the two
disagree and `reference: "code"` follows the official code. Settings the paper leaves
open come from the code in both modes; their lines are commented `official code:`,
disagreements `paper vs code:`.

This is a separate text-reasoning path beside the tensor runtime: it needs the
`reasoning` extra (transformers, peft) and imports it only when a model is built.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

from kairots.reasoning import FinTSRBench, scoring

REFERENCES = ("paper", "code")
# official code: LoRA on every attention and MLP projection of the Qwen2 blocks.
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
ASSISTANT_MARKER = "<|im_start|>assistant\n"


@dataclass(kw_only=True)
class Config:
    name: str = "FinSTaR"
    reference: str = "paper"
    # The paper table row a config reproduces (documentation only).
    row: str = ""
    # Backbone: TimeOmni-1-7B (FinSTaR) or Qwen/Qwen2.5-7B-Instruct (SFT baseline).
    model: str = "anton-hugging/TimeOmni-1-7B"
    # The FinTSR-Bench directory; an empty `train` file evaluates zero-shot.
    data_dir: str = "datasets/FinTSR-Bench"
    train: str = "train_cot"
    splits: tuple[str, ...] = ("A", "B", "C")
    # Paper Appendix C: LoRA r 32, alpha 64, lr 5e-5, 4 epochs, batch 1 with
    # gradient accumulation 16, max length 4096 (2 x L40S GPUs, bf16).
    lora_rank: int = 32
    lora_alpha: int = 64
    learning_rate: float = 5e-5
    epochs: int = 4
    batch_size: int = 1
    gradient_accumulation: int = 16
    max_length: int = 4096
    precision: str = "bf16"
    # official code (the paper is silent): LoRA dropout, AdamW decay, cosine
    # schedule with 10% warmup, class-weighted loss, gradient checkpointing.
    lora_dropout: float = 0.05
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    class_weighted: bool = True
    gradient_checkpointing: bool = True
    # official code (`inference.py`, vLLM; the paper is silent): decoding.
    temperature: float = 0.1
    top_p: float = 0.001
    repetition_penalty: float = 1.05
    max_new_tokens: int = 4096
    generation_batch_size: int = 5
    # Samples per split and task kept for evaluation, and per task for training
    # (smoke runs only); 0 keeps all.
    samples_per_task: int = 0
    train_samples_per_task: int = 0
    # A previous run directory: score its saved adapters (training skipped) and
    # keep the splits already scored in its metrics.json.
    evaluate_dir: str = ""
    # Progress lines every this many optimizer steps / generation batches.
    log_every: int = 50
    seed: int = 42
    device: str = "auto"
    runs_dir: str = "runs"
    # Hugging Face loading options (a local directory model needs neither).
    trust_remote_code: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference not in REFERENCES:
            raise ValueError(f"FinSTaR reference must be one of {REFERENCES}")
        if self.train not in ("", "train_cot", "train_ao"):
            raise ValueError("FinSTaR train is '', 'train_cot' or 'train_ao'")
        unknown = set(self.splits) - {"A", "B", "C"}
        if unknown:
            raise ValueError(f"FinTSR-Bench has test splits A, B, C, not {unknown}")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("FinSTaR precision is 'bf16' or 'fp32'")
        if min(self.epochs, self.batch_size, self.gradient_accumulation) < 1:
            raise ValueError("FinSTaR epochs and batch sizes must be >= 1")

    def evaluated_epochs(self) -> list[int]:
        """paper vs code: the paper reports the model after its 4 epochs; the code's
        evaluation script scores the adapter of every epoch (`06_cot_eval_*.sh`
        loops EVAL_EP 1-4), reported here per epoch with no selection."""
        if not self.train:
            return [0]
        if self.reference == "code":
            return list(range(1, self.epochs + 1))
        return [self.epochs]


def _dtype(*, config: Config) -> torch.dtype:
    return torch.bfloat16 if config.precision == "bf16" else torch.float32


def _device(*, config: Config) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def load_backbone(*, config: Config, adapter: str | Path | None = None) -> Any:
    """(model, tokenizer); `adapter` loads a saved LoRA adapter on the backbone."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise ImportError(
            "FinSTaR needs the `reasoning` extra: pip install -e .[reasoning]"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, trust_remote_code=config.trust_remote_code
    )
    # official code: an absent pad token falls back to end-of-sequence.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=_dtype(config=config),
        trust_remote_code=config.trust_remote_code,
    )
    if adapter is not None:
        from peft import PeftModel  # pyright: ignore[reportMissingImports]

        model = PeftModel.from_pretrained(model, str(adapter))
    return model.to(_device(config=config)), tokenizer


def encode(
    *, tokenizer: Any, prompt: str, target: str, max_length: int
) -> dict[str, torch.Tensor]:
    """official code (`FinTSRDatasetV2`): the chat template over the user prompt and
    the assistant target, truncated to `max_length`; the loss covers the tokens after
    the first assistant marker."""
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    input_ids = torch.tensor(
        tokenizer(text, max_length=max_length, truncation=True)["input_ids"]
    )
    labels = input_ids.clone()
    marker = tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)
    end = 0
    for start in range(len(input_ids) - len(marker) + 1):
        if input_ids[start : start + len(marker)].tolist() == marker:
            end = start + len(marker)
            break
    labels[:end] = -100
    return {"input_ids": input_ids, "labels": labels}


def _collate(
    *, items: list[dict[str, torch.Tensor]], pad: int
) -> dict[str, torch.Tensor]:
    """official code: right padding to the longest item of the batch."""
    length = max(len(item["input_ids"]) for item in items)

    def padded(*, tensor: torch.Tensor, value: int) -> torch.Tensor:
        return functional.pad(tensor, (0, length - len(tensor)), value=value)

    return {
        "input_ids": torch.stack(
            [padded(tensor=item["input_ids"], value=pad) for item in items]
        ),
        "attention_mask": torch.stack(
            [
                padded(tensor=torch.ones_like(item["input_ids"]), value=0)
                for item in items
            ]
        ),
        "labels": torch.stack(
            [padded(tensor=item["labels"], value=-100) for item in items]
        ),
    }


def sample_loss(
    *, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None
) -> torch.Tensor:
    """official code (`--class_weighted`): each sample's mean token cross-entropy,
    times its class weight, averaged over the batch; unweighted, the mean over all
    target tokens (the Hugging Face loss)."""
    shift_logits = logits[..., :-1, :].float()
    shift_labels = labels[..., 1:]
    per_token = functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    mask = (shift_labels != -100).float()
    if weights is None:
        return (per_token * mask).sum() / mask.sum().clamp(min=1)
    per_sample = (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return (per_sample * weights).mean()


def train(
    *, config: Config, samples: list[FinTSRBench.Sample], output_dir: Path
) -> list[Path]:
    """LoRA SFT; saves the adapter after every epoch and returns their directories."""
    from peft import LoraConfig, TaskType, get_peft_model  # pyright: ignore[reportMissingImports]
    from transformers import get_cosine_schedule_with_warmup  # pyright: ignore[reportMissingImports]

    torch.manual_seed(config.seed)
    model, tokenizer = load_backbone(config=config)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(TARGET_MODULES),
        ),
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    device = _device(config=config)
    items = [
        encode(
            tokenizer=tokenizer,
            prompt=sample.prompt,
            target=sample.target or "",
            max_length=config.max_length,
        )
        for sample in samples
    ]
    weights = (
        torch.tensor(FinTSRBench.class_weights(samples=samples))
        if config.class_weighted
        else None
    )
    # positional: the torch.optim learning-rate argument
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batches = math.ceil(len(items) / config.batch_size)
    # official code: total = batches x epochs // accumulation optimizer steps.
    total = max(1, batches * config.epochs // config.gradient_accumulation)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total * config.warmup_fraction), total
    )
    generator = torch.Generator().manual_seed(config.seed)
    saved = []
    model.train()
    started, step, running = time.perf_counter(), 0, 0.0
    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(len(items), generator=generator)
        parts = order.split(config.batch_size)
        for index, part in enumerate(parts, start=1):
            batch = _collate(
                items=[items[position] for position in part.tolist()],
                pad=tokenizer.pad_token_id,
            )
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            loss = sample_loss(
                logits=logits,
                labels=batch["labels"].to(device),
                weights=None if weights is None else weights[part].to(device),
            )
            (loss / config.gradient_accumulation).backward()
            running += loss.item() / config.gradient_accumulation
            # official code (accelerate): step every `gradient_accumulation`
            # micro-batches and at the end of each pass.
            if index % config.gradient_accumulation == 0 or index == len(parts):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % config.log_every == 0 or step == total:
                    elapsed = time.perf_counter() - started
                    print(
                        f"train epoch {epoch} step {step}/{total} loss "
                        f"{running:.4f} lr {scheduler.get_last_lr()[0]:.3e} "
                        f"elapsed {elapsed / 3600:.2f} h eta "
                        f"{elapsed / step * (total - step) / 3600:.2f} h",
                        flush=True,
                    )
                running = 0.0
        directory = output_dir / f"epoch_{epoch}"
        model.save_pretrained(str(directory))
        tokenizer.save_pretrained(str(directory))
        saved.append(directory)
    return saved


def generate(
    *, config: Config, model: Any, tokenizer: Any, prompts: list[str]
) -> list[str]:
    """Answers with the official decoding (temperature 0.1, top-p 0.001, repetition
    penalty 1.05); prompts use the chat template the model was trained with."""
    device = _device(config=config)
    tokenizer.padding_side = "left"
    model.eval()
    texts: list[str] = []
    for start in range(0, len(prompts), config.generation_batch_size):
        chats = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts[start : start + config.generation_batch_size]
        ]
        encoded = tokenizer(chats, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts += tokenizer.batch_decode(
            output[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        batch = start // config.generation_batch_size + 1
        if batch % config.log_every == 0:
            print(f"generate {len(texts)}/{len(prompts)}", flush=True)
    return texts


def _subset(
    *, samples: list[FinTSRBench.Sample], per_task: int
) -> list[FinTSRBench.Sample]:
    if not per_task:
        return samples
    counts: dict[str, int] = {}
    kept = []
    for sample in samples:
        if counts.get(sample.task, 0) < per_task:
            counts[sample.task] = counts.get(sample.task, 0) + 1
            kept.append(sample)
    return kept


def run(*, config: Config) -> dict[str, Any]:
    """Fine-tune (unless zero-shot), then generate and score each test split."""
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    data = Path(config.data_dir)
    adapters: dict[int, Path | None] = {0: None}
    metrics: dict[str, Any] = {}
    if config.evaluate_dir:
        run_dir = Path(config.evaluate_dir)
        if config.train:
            adapters = {
                epoch: run_dir / "lora" / f"epoch_{epoch}"
                for epoch in range(1, config.epochs + 1)
            }
        if (run_dir / "metrics.json").is_file():
            metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(config.runs_dir) / f"FinSTaR-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )
        if config.train:
            samples = _subset(
                samples=FinTSRBench.load(
                    path=data / FinTSRBench.FILES[config.train],
                    answers_only=config.train == "train_ao",
                ),
                per_task=config.train_samples_per_task,
            )
            trained = train(config=config, samples=samples, output_dir=run_dir / "lora")
            adapters = dict(enumerate(trained, start=1))
    for epoch in config.evaluated_epochs():
        labels = {
            split: f"epoch_{epoch}/{split}" if config.train else split
            for split in config.splits
        }
        if all(label in metrics for label in labels.values()):
            continue
        model, tokenizer = load_backbone(config=config, adapter=adapters[epoch])
        for split in config.splits:
            if labels[split] in metrics:
                continue
            print(f"evaluate {labels[split]}", flush=True)
            samples = _subset(
                samples=FinTSRBench.load(path=data / FinTSRBench.FILES[split]),
                per_task=config.samples_per_task,
            )
            texts = generate(
                config=config,
                model=model,
                tokenizer=tokenizer,
                prompts=[sample.prompt for sample in samples],
            )
            label = labels[split]
            with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as file:
                for sample, text in zip(samples, texts, strict=True):
                    file.write(
                        json.dumps(
                            {
                                "run": label,
                                "task": sample.task,
                                "answer": sample.answer,
                                "output": text,
                            }
                        )
                        + "\n"
                    )
            metrics[label] = scoring.score(
                results=[
                    (sample.task, sample.answer, text)
                    for sample, text in zip(samples, texts, strict=True)
                ]
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
    }


def config_from(*, values: dict[str, Any], overrides: dict[str, Any]) -> Config:
    known = {item.name for item in fields(Config)}
    merged = {**values, **overrides}
    unknown = set(merged) - known
    if unknown:
        raise ValueError(f"unknown FinSTaR settings: {sorted(unknown)}")
    if "splits" in merged:
        merged["splits"] = tuple(merged["splits"])
    return Config(**merged)
