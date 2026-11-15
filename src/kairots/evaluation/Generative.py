"""Sample-based scoring of a generative model (`Generation`).

Per client, the real windows are every `output_length`-th window of the phase
(non-overlapping, TimeDP's evaluation stride), the model samples as many windows from
noise, and each metric compares the two on scaled values. A metric that NEEDS
`train` (DCR) also reads the client's train windows. The phase loss is the denoising
MSE on the real windows. Noise and samples restart from the task's `seed` for every
client, and each client samples with its own parameters when the strategy keeps
per-client models.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from kairots.evaluation.base import Scoring
from kairots.registry import get, info


@dataclass(kw_only=True)
class Generative:
    TASKS = ("Generation",)

    @dataclass(kw_only=True)
    class Config(Scoring.Config):
        name: str = "Generative"
        metrics: dict[str, tuple[str, ...]] = field(
            default_factory=lambda: {"validation": (), "test": ()}
        )

    def __init__(self, *, config: Config, loss_function: Any) -> None:
        self.config = config
        self.loss_function = loss_function

    @staticmethod
    def _windows(
        *,
        values: Tensor,
        windows: list[tuple[int, int]],
        experiment: Any,
        view: Any,
    ) -> Tensor:
        """The windows' values in order, (windows, steps, channels)."""
        input_length, output_length, offset_length = experiment.lengths()
        return torch.cat(
            [
                batch_y
                for _, batch_y in view.batches(
                    values=values,
                    windows=windows,
                    input_length=input_length,
                    output_length=output_length,
                    offset_length=offset_length,
                    batch_size=experiment.data.source.batch_size,
                    shuffle=False,
                    generator=torch.Generator(),
                )
            ]
        )

    def evaluate(
        self,
        *,
        phase: str,
        model: nn.Module,
        clients: list[tuple[Tensor, dict[str, list[tuple[int, int]]]]],
        parameters: list[dict[str, Tensor]] | None = None,
        experiment: Any,
        view: Any,
        task: Any,
        device: torch.device,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        metric_names = self.config.metrics.get(phase, ())
        metric_types = {
            name.lower(): get(category="metrics", name=name) for name in metric_names
        }
        reads_train = {
            name.lower(): "train" in info(category="metrics", name=name)["NEEDS"]
            for name in metric_names
        }
        pooled = {
            name: metric_type(config=metric_type.Config())
            for name, metric_type in metric_types.items()
        }
        _, output_length, _ = experiment.lengths()
        batch_size = experiment.data.source.batch_size
        loss_name = self.config.loss.lower()
        loss_total = torch.zeros((), device=device)
        count_total = torch.zeros((), device=device)
        rows: list[dict[str, float]] = []
        model.eval()
        with torch.no_grad():
            for index, (values, splits) in enumerate(clients):
                if parameters is not None:
                    model.load_state_dict(parameters[index])
                windows = splits[phase][::output_length]
                if not windows:
                    rows.append({})
                    continue
                real = self._windows(
                    values=values, windows=windows, experiment=experiment, view=view
                )
                generator = torch.Generator().manual_seed(task.config.seed)
                loss_sum = torch.zeros((), device=device)
                count = 0
                for batch_y in real.split(batch_size):
                    model_input, noise = task.prepare(
                        batch_x=batch_y[:, :0],
                        batch_y=batch_y,
                        output_length=output_length,
                        generator=generator,
                    )
                    noise = noise.to(device=device)
                    prediction = model(batch_x=model_input.to(device=device))
                    loss_sum += self.loss_function(
                        prediction=prediction, target=noise
                    ) * (noise.numel())
                    count += noise.numel()
                row = {loss_name: float((loss_sum / count).item())}
                loss_total += loss_sum
                count_total += count
                if metric_types:
                    generated = task.sample(
                        model=model,
                        count=len(real),
                        shape=tuple(real.shape[1:]),
                        generator=generator,
                        device=device,
                    ).cpu()
                    train = (
                        self._windows(
                            values=values,
                            windows=splits["train"],
                            experiment=experiment,
                            view=view,
                        )
                        if any(reads_train.values())
                        else None
                    )
                    for name, metric_type in metric_types.items():
                        metric = metric_type(config=metric_type.Config())
                        metric.update(
                            prediction=generated,
                            target=real,
                            **({"train": train} if reads_train[name] else {}),
                        )
                        row[name] = metric.compute()
                        pooled[name].merge(other=metric)
                rows.append(row)
        scored = [row for row in rows if row]
        if not scored:
            raise ValueError(f"{phase} evaluation has no eligible windows")
        metrics = {f"pooled_{loss_name}": float((loss_total / count_total).item())}
        metrics.update({f"pooled_{name}": pooled[name].compute() for name in pooled})
        for name in scored[0]:
            metrics[f"mean_client_{name}"] = sum(row[name] for row in scored) / len(
                scored
            )
        return metrics, rows
