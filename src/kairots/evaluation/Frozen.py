"""Frozen global-model evaluation on the train, validation and test populations."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.evaluation.base import Scoring
from kairots.metrics._mean import add
from kairots.metrics.MASE import seasonal_scales
from kairots.metrics.Return import Return
from kairots.registry import get


@dataclass(kw_only=True)
class Frozen:
    GROUPS = True

    @dataclass(kw_only=True)
    class Config(Scoring.Config):
        name: str = "Frozen"
        # Scores are on scaled values. `inverse` scores the test phase in raw units:
        # it undoes each client's train-fitted scaler on predictions, targets and
        # history (TSLib `--inverse`), so a channel with a large range counts at its
        # real size. Validation stays scaled, so checkpoint selection does not
        # change. Class labels are never rescaled.
        inverse: bool = False
        # Epochs each seen client fine-tunes a copy of the model it is scored with,
        # on its own train windows, before the final round is scored (`finetuned_*`);
        # the copy is discarded and never sent. 0 turns it off.
        finetune_epochs: int = 0

        def __post_init__(self) -> None:
            if self.finetune_epochs < 0:
                raise ValueError("finetune_epochs must be >= 0")

    def __init__(self, *, config: Config, loss_function: Any) -> None:
        self.config = config
        self.loss_function = loss_function

    def start_client(self, *, phase: str, model: nn.Module) -> None:
        """Hook before each client's windows; the frozen protocol keeps no state."""

    def end_client(self, *, phase: str, model: nn.Module) -> None:
        """Hook after each client's last window is scored."""

    def predict(
        self,
        *,
        phase: str,
        model: nn.Module,
        batch_x: Tensor,
        windows: list[tuple[int, int]],
    ) -> Tensor:
        """One batch's prediction; it sees inputs only, never the batch's targets.

        `windows` are the batch's (start, target) positions in the client's values.
        """
        return model(batch_x=batch_x)

    def observe(
        self,
        *,
        phase: str,
        batch_x: Tensor,
        batch_y: Tensor,
        windows: list[tuple[int, int]],
    ) -> None:
        """Hook after a batch is scored, when its targets exist but are not yet due."""

    def evaluate(
        self,
        *,
        phase: str,
        model: nn.Module,
        clients: list[tuple[Tensor, dict[str, list[tuple[int, int]]]]],
        parameters: list[dict[str, Tensor]] | None = None,
        scales: list[tuple[Tensor, Tensor]] | None = None,
        models: list[nn.Module] | None = None,
        experiments: list[Any] | None = None,
        experiment: Any,
        view: Any,
        task: Any,
        device: torch.device,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        """Score each client's `phase` windows and the pooled population.

        `clients` holds each client's values and windows of every split; rows align
        with `clients`. `parameters`, when given, holds each client's own model
        parameters (personalized or local-only methods); otherwise `model` is scored
        as is. `scales` holds each client's scaler (mean, scale), needed by `inverse`.
        `models` and `experiments`, when given, hold each client's model and its
        group's experiment (client groups, model-heterogeneous strategies).
        """
        metric_types = {
            name.lower(): get(category="metrics", name=name)
            for name in self.config.metrics.get(phase, ())
        }
        metric_contexts = {}
        for name, metric_type in metric_types.items():
            prepare_context = getattr(metric_type, "prepare_context", None)
            if prepare_context is not None:
                context_name = getattr(metric_type, "CONTEXT", None)
                if context_name is None:
                    raise TypeError(
                        f"{metric_type.__name__}.prepare_context requires CONTEXT"
                    )
                metric_contexts[name] = (
                    context_name,
                    prepare_context(experiment=experiment),
                )
        pooled = {
            name: metric_type(config=metric_type.Config())
            for name, metric_type in metric_types.items()
        }
        loss_name = self.config.loss.lower()
        loss_total = torch.zeros((), device=device)
        count_total = torch.zeros((), device=device)
        rows: list[dict[str, float]] = []
        with torch.no_grad():
            for index, (values, splits) in enumerate(clients):
                if models is not None:
                    model = models[index]
                if experiments is not None:
                    experiment = experiments[index]
                    # A group's task and view resolve its own channels.
                    task = get(category="tasks", name=experiment.task.name)(
                        config=experiment.task
                    )
                    view = get(category="data/views", name=experiment.data.view.name)(
                        config=experiment.data.view
                    )
                input_length, output_length, offset_length = experiment.lengths()
                model.eval()
                if parameters is not None:
                    model.load_state_dict(parameters[index])
                self.start_client(phase=phase, model=model)
                windows = splits[phase]
                restore = (
                    self.config.inverse
                    and phase == "test"
                    and getattr(task, "HEAD", "")
                    in ("Forecasting", "Quantile", "Gaussian", "Samples")
                )
                if restore and scales is None:
                    raise ValueError("inverse scoring needs each client's scaler")
                # Columns of the client values that are forecast targets.
                target_columns = getattr(task, "target_index", slice(None))
                mean, scale = (
                    (
                        scales[index][0][target_columns].to(device),
                        scales[index][1][target_columns].to(device),
                    )
                    if restore and scales is not None
                    else (torch.zeros(()), torch.ones(()))
                )
                client_metrics = {
                    name: metric_type(config=metric_type.Config())
                    for name, metric_type in metric_types.items()
                }
                # A metric that NEEDS the seasonal scale (MASE) gets each window's,
                # from the client's series before its target, in window order.
                seasonal = None
                if windows and any(
                    "seasonal_scale" in metric.NEEDS
                    for metric in client_metrics.values()
                ):
                    series = values[:, target_columns]
                    if restore:
                        series = series.to(device) * scale + mean
                    seasonal = seasonal_scales(
                        series=series,
                        target_rows=[target for _, target in windows],
                        seasonality=experiment.data.source.seasonality,
                    ).to(device)
                loss_sum = torch.zeros((), device=device)
                count = torch.zeros((), device=device)
                # Scoring masks restart from the task's seed for every client and
                # round, so each evaluation hides the same positions.
                mask_generator = torch.Generator().manual_seed(
                    getattr(task.config, "seed", 0)
                )
                batch_size = experiment.data.source.batch_size
                for position, (batch_x, batch_y) in enumerate(
                    view.batches(
                        values=values,
                        windows=windows,
                        input_length=input_length,
                        output_length=output_length,
                        offset_length=offset_length,
                        batch_size=batch_size,
                        shuffle=False,
                        generator=torch.Generator(),
                    )
                ):
                    batch_x, batch_y = task.prepare(
                        batch_x=batch_x,
                        batch_y=batch_y,
                        output_length=output_length,
                        generator=mask_generator,
                    )
                    batch_x = batch_x.to(device=device)
                    batch_y = batch_y.to(device=device)
                    first = position * batch_size
                    batch_windows = windows[first : first + batch_size]
                    prediction = self.predict(
                        phase=phase, model=model, batch_x=batch_x, windows=batch_windows
                    )
                    scored_prediction, scored_target = prediction, batch_y
                    if restore:
                        # Quantile predictions carry a last axis of levels.
                        extra = prediction.dim() - batch_y.dim()
                        scored_prediction = prediction * scale.reshape(
                            scale.shape + (1,) * extra
                        ) + mean.reshape(mean.shape + (1,) * extra)
                        scored_target = batch_y * scale + mean
                        if task.HEAD == "Gaussian":
                            scored_prediction = torch.stack(
                                [
                                    prediction[..., 0] * scale + mean,
                                    prediction[..., 1] * scale.abs(),
                                ],
                                dim=-1,
                            )
                    # Non-finite targets are unscored positions (hidden-free or
                    # natively missing); a loss averages over the scored ones.
                    scored = scored_target.isfinite().sum()
                    batch_loss = self.loss_function(
                        prediction=scored_prediction, target=scored_target
                    )
                    loss_sum += batch_loss * scored
                    count += scored
                    for name, metric in client_metrics.items():
                        if (
                            type(self.loss_function) is Return
                            and type(metric) is Return
                        ):
                            add(
                                metric=metric,
                                total=-batch_loss * len(scored_target),
                                count=len(scored_target),
                            )
                            continue
                        metric.update(
                            prediction=scored_prediction,
                            target=scored_target,
                            **(
                                {"seasonal_scale": seasonal[first : first + batch_size]}
                                if seasonal is not None
                                and "seasonal_scale" in metric.NEEDS
                                else {}
                            ),
                            **(
                                {metric_contexts[name][0]: metric_contexts[name][1]}
                                if name in metric_contexts
                                else {}
                            ),
                        )
                    self.observe(
                        phase=phase,
                        batch_x=batch_x,
                        batch_y=batch_y,
                        windows=batch_windows,
                    )
                self.end_client(phase=phase, model=model)
                if not windows:
                    rows.append({})
                    continue
                # One host sync per client; batch them if clients number
                # in the thousands.
                row = {loss_name: float((loss_sum / count).item())}
                for name, metric in client_metrics.items():
                    row[name] = metric.compute()
                    pooled[name].merge(other=metric)
                rows.append(row)
                loss_total += loss_sum
                count_total += count
        scored = [row for row in rows if row]
        if not scored:
            raise ValueError(f"{phase} evaluation has no eligible windows")
        metrics = {f"pooled_{loss_name}": float((loss_total / count_total).item())}
        metrics.update({f"pooled_{name}": pooled[name].compute() for name in pooled})
        metrics.update(
            {
                f"mean_client_{name}": sum(row[name] for row in scored) / len(scored)
                for name in scored[0]
            }
        )
        return metrics, rows
