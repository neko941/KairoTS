"""Frozen reconstruction scoring with a calibrated anomaly threshold (TSLib).

Per client, each step's anomaly score is the channel-mean squared reconstruction
error of the frozen global model. The threshold is the `100 - anomaly_ratio`
percentile of the calibration scores, then test steps above it are predicted
anomalous and scored against the labels:

- `calibration="validation"` (default): validation scores only;
- `calibration="train"`: train scores only, for runs without a validation split;
- `calibration="train_test"` (TSLib): train scores plus unlabeled test scores, for
  comparison with TSLib's numbers; the threshold then depends on the test data.

Test labels never set the threshold. Each client calibrates on its own scores
(with its own parameters when the strategy keeps per-client models).
Train and validation phases report the reconstruction loss only. Test windows tile
the test fold (`Predefined.series`), so the flattened scores form its timeline.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from kairots.evaluation.base import Scoring
from kairots.registry import get


@dataclass(kw_only=True)
class Calibrated:
    @dataclass(kw_only=True)
    class Config(Scoring.Config):
        # Phases whose scores set the threshold, per calibration.
        PHASES = {
            "train": ("train",),
            "validation": ("validation",),
            "train_test": ("train", "test"),
        }

        name: str = "Calibrated"
        metrics: dict[str, tuple[str, ...]] = field(
            default_factory=lambda: {"validation": (), "test": ("F1", "AdjustedF1")}
        )
        # Percent of calibration steps above the threshold (TSLib SMD: 0.5).
        anomaly_ratio: float = 0.5
        calibration: str = "validation"

        def __post_init__(self) -> None:
            if self.calibration not in self.PHASES:
                raise ValueError(f"calibration must be one of {sorted(self.PHASES)}")
            if not 0.0 < self.anomaly_ratio < 100.0:
                raise ValueError("anomaly_ratio is a percent in (0, 100)")

    def __init__(self, *, config: Config, loss_function: Any) -> None:
        self.config = config
        self.loss_function = loss_function

    def _scores(
        self,
        *,
        model: nn.Module,
        values: Tensor,
        windows: list[tuple[int, int]],
        experiment: Any,
        view: Any,
        task: Any,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-step scores and labels in window order, plus loss sum and count."""
        input_length, output_length, offset_length = experiment.lengths()
        scores: list[Tensor] = []
        labels: list[Tensor] = []
        loss_sum = torch.zeros((), device=device)
        count = torch.zeros((), device=device)
        for batch_x, batch_y in view.batches(
            values=values,
            windows=windows,
            input_length=input_length,
            output_length=output_length,
            offset_length=offset_length,
            batch_size=experiment.data.source.batch_size,
            shuffle=False,
            generator=torch.Generator(),
        ):
            labels.append(batch_y[..., -1].reshape(-1))
            batch_x, batch_y = task.prepare(
                batch_x=batch_x, batch_y=batch_y, output_length=output_length
            )
            batch_x = batch_x.to(device=device)
            batch_y = batch_y.to(device=device)
            prediction = model(batch_x=batch_x)
            loss_sum += self.loss_function(prediction=prediction, target=batch_y) * (
                batch_y.numel()
            )
            count += batch_y.numel()
            scores.append((prediction - batch_y).square().mean(dim=-1).reshape(-1))
        empty = torch.zeros(0)
        return (
            torch.cat(scores).cpu() if scores else empty,
            torch.cat(labels) if labels else empty,
            loss_sum,
            count,
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
        metric_types = {
            name.lower(): get(category="metrics", name=name)
            for name in self.config.metrics.get(phase, ())
        }
        pooled = {
            name: metric_type(config=metric_type.Config())
            for name, metric_type in metric_types.items()
        }
        loss_name = self.config.loss.lower()
        loss_total = torch.zeros((), device=device)
        count_total = torch.zeros((), device=device)
        rows: list[dict[str, float]] = []
        model.eval()
        with torch.no_grad():
            for index, (values, splits) in enumerate(clients):
                if parameters is not None:
                    model.load_state_dict(parameters[index])
                if not splits[phase]:
                    rows.append({})
                    continue
                scores, labels, loss_sum, count = self._scores(
                    model=model,
                    values=values,
                    windows=splits[phase],
                    experiment=experiment,
                    view=view,
                    task=task,
                    device=device,
                )
                row = {loss_name: float((loss_sum / count).item())}
                loss_total += loss_sum
                count_total += count
                if metric_types:
                    calibration_phases = self.config.PHASES[self.config.calibration]
                    if not all(splits[name] for name in calibration_phases):
                        raise ValueError(
                            f"calibration={self.config.calibration!r} needs "
                            f"{calibration_phases} windows; set "
                            "data.split.validation_ratio or calibration='train'"
                        )
                    calibration = torch.cat(
                        [
                            scores
                            if name == phase
                            else self._scores(
                                model=model,
                                values=values,
                                windows=splits[name],
                                experiment=experiment,
                                view=view,
                                task=task,
                                device=device,
                            )[0]
                            for name in calibration_phases
                        ]
                    )
                    threshold = torch.quantile(
                        calibration.double(), 1 - self.config.anomaly_ratio / 100
                    )
                    predicted = scores.double() > threshold
                    for name, metric_type in metric_types.items():
                        metric = metric_type(config=metric_type.Config())
                        metric.update(prediction=predicted, target=labels)
                        row[name] = metric.compute()
                        pooled[name].merge(other=metric)
                rows.append(row)
        scored = [row for row in rows if row]
        if not scored:
            raise ValueError(f"{phase} evaluation has no eligible windows")
        metrics = {f"pooled_{loss_name}": float((loss_total / count_total).item())}
        metrics.update({f"pooled_{name}": pooled[name].compute() for name in pooled})
        for name in scored[0]:
            # An undefined client score (NaN F1: no anomalies, no alarms) is skipped.
            defined = [row[name] for row in scored if row[name] == row[name]]
            metrics[f"mean_client_{name}"] = (
                sum(defined) / len(defined) if defined else float("nan")
            )
        return metrics, rows
