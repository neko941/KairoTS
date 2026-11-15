"""TAFAS: test-time adaptation of a frozen forecaster with partially observed targets.

Kim et al., "Battling the Non-stationarity in Time Series Forecasting via Test-time
Adaptation", AAAI 2025, arXiv:2501.04970. Original source: kimanki/TAFAS,
`tta/tafas.py`.

The test windows of each client form a stride-1 stream. Train and validation phases
are scored frozen. On the test phase the protocol first scores the frozen model
(`frozen_*`), then streams the windows in time order. Periodicity-aware adaptation
scheduling reads the dominant period `p` of a mini-batch's first look-back window
and groups `p + 1` consecutive windows. Once the last of them has arrived, the
gated calibration modules before and after the frozen backbone take `steps`
optimizer steps on the first `p` observed targets of the mini-batch's first window,
and the full targets of the most recent earlier mini-batch whose targets have all
arrived. Their mean squared losses are summed for one optimizer update. The
predictions of the mini-batch are then recomputed, and each window's
steps that were still unobserved at that time are replaced by the adapted values.
A target row is read only after the stream has observed its time step, so no
update or prediction depends on a value that has not arrived. Only the calibration
modules adapt; the backbone and its state are untouched, and each client streams
from a fresh identity calibration.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.evaluation.Frozen import Frozen
from kairots.learners.base import Learning
from kairots.optimizers.base import UpdateBudget
from kairots.registry import get

# The original code's period when the dominant frequency is the zero bin.
FALLBACK_PERIOD = 24


class Calibration(nn.Module):
    """Variable-wise gated temporal calibration, identity at initialization.

    `x + tanh(gating) * (W x + bias)` with one `[length, length]` map per channel.
    """

    def __init__(self, *, length: int, channels: int, gating_init: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(length, length, channels))
        self.gating = nn.Parameter(torch.full((channels,), gating_init))
        self.bias = nn.Parameter(torch.zeros(length, channels))

    def forward(self, *, batch: Tensor) -> Tensor:
        calibrated = torch.einsum("biv,iov->bov", batch, self.weight) + self.bias
        return batch + torch.tanh(self.gating) * calibrated


@dataclass(kw_only=True)
class TAFAS(Frozen):
    TASKS = ("Forecasting",)
    REQUIRES = {
        "task.name": ("Forecasting",),
        "groups": ((),),
        "client_model": (None,),
        "task.calendar": ((),),
        "task.target_channels": ((),),
    }

    @dataclass(kw_only=True)
    class Config(Frozen.Config, Learning.Config, UpdateBudget.Config):
        name: str = "TAFAS"
        # The original scripts' DLinear settings: Adam, learning rate 1e-3.
        learning_rate: float = 1e-3
        # Initial value of the gating vector before tanh.
        gating_init: float = 0.01

        def __post_init__(self) -> None:
            if self.steps < 0:
                raise ValueError("steps must be non-negative")

    def __init__(self, *, config: Config, loss_function: Any) -> None:
        super().__init__(config=config, loss_function=loss_function)
        self.config: TAFAS.Config = config
        self.collecting = False
        self.replaying = False
        self.client = -1
        self.cursor = 0
        self.updates = 0
        # Per client: its test batches (inputs, targets, windows), then its
        # predictions once the stream has been walked.
        self.streams: list[list[tuple[Tensor, Tensor, list[tuple[int, int]]]]] = []
        self.predictions: list[Tensor] = []
        self.final_times: list[list[int]] = []

    def start_client(self, *, phase: str, model: nn.Module) -> None:
        self.client += 1
        self.cursor = 0
        if self.collecting:
            self.streams.append([])

    def observe(
        self,
        *,
        phase: str,
        batch_x: Tensor,
        batch_y: Tensor,
        windows: list[tuple[int, int]],
    ) -> None:
        if self.collecting:
            self.streams[self.client].append((batch_x, batch_y, windows))

    def predict(
        self,
        *,
        phase: str,
        model: nn.Module,
        batch_x: Tensor,
        windows: list[tuple[int, int]],
    ) -> Tensor:
        if not self.replaying:
            return super().predict(
                phase=phase, model=model, batch_x=batch_x, windows=windows
            )
        rows = self.predictions[self.client][self.cursor : self.cursor + len(batch_x)]
        self.cursor += len(batch_x)
        return rows

    @staticmethod
    def period(*, window: Tensor) -> int:
        """Dominant period of a `[input_length, channels]` look-back window."""
        amplitude = torch.fft.rfft(window - window.mean(dim=0), dim=0).abs()
        channel = int((amplitude**2).mean(dim=0).argmax())
        frequency = int(amplitude[:, channel].argmax())
        return (
            (window.shape[0] + frequency - 1) // frequency
            if frequency
            else FALLBACK_PERIOD
        )

    @staticmethod
    def revealed(
        *,
        target: Tensor,
        windows: list[tuple[int, int]],
        first: int,
        last: int,
        rows: int,
        observed: int,
    ) -> Tensor:
        """First `rows` targets of windows `first..last`, once all are observed."""
        if windows[last - 1][1] + rows > observed:
            raise ValueError("a target was read before the stream observed it")
        return target[first:last, :rows]

    def stream(
        self,
        *,
        model: nn.Module,
        batch_x: Tensor,
        batch_y: Tensor,
        windows: list[tuple[int, int]],
    ) -> tuple[Tensor, list[int]]:
        """Adapted predictions of one client's stride-1 test stream, and the time
        step count the stream had observed when each window's prediction was final."""
        if batch_x.shape[-1] != batch_y.shape[-1]:
            raise ValueError("TAFAS needs forecast channels equal to input channels")
        if any(
            later[0] - earlier[0] != 1 or later[1] - earlier[1] != 1
            for earlier, later in zip(windows, windows[1:], strict=False)
        ):
            raise ValueError("TAFAS needs consecutive test windows (stride 1)")
        count, input_length, channels = batch_x.shape
        output_length = batch_y.shape[1]
        config = self.config
        before = Calibration(
            length=input_length, channels=channels, gating_init=config.gating_init
        ).to(batch_x.device)
        after = Calibration(
            length=output_length, channels=channels, gating_init=config.gating_init
        ).to(batch_x.device)
        parameters = [*before.parameters(), *after.parameters()]
        optimizer_type = get(category="optimizers", name=config.optimizer.name)
        optimizer = optimizer_type(config=config.optimizer).build(
            parameters=parameters, learning_rate=config.learning_rate, fused=False
        )

        def forward(*, batch: Tensor) -> Tensor:
            return after(batch=model(batch_x=before(batch=batch)))

        def fit(
            *, batch: Tensor, target: Tensor, full: tuple[Tensor, Tensor] | None
        ) -> Tensor:
            """`steps` updates on the leading windows and rows of `target`; returns
            the prediction of the last forward pass, taken before its update."""
            prediction = batch
            for _ in range(config.steps):
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(batch=batch)
                loss = (
                    (prediction[: target.shape[0], : target.shape[1]] - target)
                    .pow(2)
                    .mean()
                )
                if full is not None:
                    loss = loss + (forward(batch=full[0]) - full[1]).pow(2).mean()
                for parameter, gradient in zip(
                    parameters, torch.autograd.grad(loss, parameters), strict=True
                ):
                    parameter.grad = gradient
                optimizer.step()
                self.updates += 1
            return prediction.detach()

        predictions = torch.empty_like(batch_y)
        final_times = [0] * count
        waiting: list[tuple[int, int]] = []
        history: tuple[int, int] | None = None
        position = 0
        with torch.enable_grad():
            while position < count:
                period = self.period(window=batch_x[position])
                end = min(position + period + 1, count)
                observed = windows[end - 1][0] + input_length
                while (
                    waiting
                    and windows[waiting[0][1] - 1][1] + output_length <= observed
                ):
                    history = waiting.pop(0)
                window = batch_x[position:end]
                if windows[position][1] + period <= observed:
                    full = None
                    if history is not None:
                        first, last = history
                        full = (
                            batch_x[first:last],
                            self.revealed(
                                target=batch_y,
                                windows=windows,
                                first=first,
                                last=last,
                                rows=output_length,
                                observed=observed,
                            ),
                        )
                    prediction = fit(
                        batch=window,
                        full=full,
                        target=self.revealed(
                            target=batch_y,
                            windows=windows,
                            first=position,
                            last=position + 1,
                            rows=period,
                            observed=observed,
                        ),
                    )
                    with torch.no_grad():
                        adapted = forward(batch=window)
                    # Replace only predictions whose targets have not arrived.
                    for offset in range(end - position):
                        prediction[offset, period - offset :] = adapted[
                            offset, period - offset :
                        ]
                else:
                    # The stream ended before the first window's partial targets
                    # arrived: no update, and no unobserved target is read.
                    with torch.no_grad():
                        prediction = forward(batch=window)
                predictions[position:end] = prediction
                final_times[position:end] = [observed] * (end - position)
                waiting.append((position, end))
                position = end
        return predictions, final_times

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
        if models is not None or experiments is not None:
            raise ValueError(f"{type(self).__name__} scores one model and experiment")
        arguments = {
            "phase": phase,
            "model": model,
            "clients": clients,
            "parameters": parameters,
            "scales": scales,
            "experiment": experiment,
            "view": view,
            "task": task,
            "device": device,
        }
        if phase != "test":
            return super().evaluate(**arguments)
        # Pass one scores the frozen model and keeps each client's test batches.
        self.collecting, self.client, self.streams = True, -1, []
        try:
            frozen, rows = super().evaluate(**arguments)
        finally:
            self.collecting = False
        source = {key: value.clone() for key, value in model.state_dict().items()}
        # Every client starts from its own parameters, or from the shared model.
        arguments["parameters"] = parameters or [source] * len(clients)
        self.updates, self.predictions, self.final_times = 0, [], []
        try:
            for index, batches in enumerate(self.streams if self.config.steps else []):
                model.load_state_dict(arguments["parameters"][index])
                model.eval()
                if not batches:
                    self.predictions.append(torch.empty(0))
                    self.final_times.append([])
                    continue
                prediction, final_times = self.stream(
                    model=model,
                    batch_x=torch.cat([batch[0] for batch in batches]),
                    batch_y=torch.cat([batch[1] for batch in batches]),
                    windows=[window for batch in batches for window in batch[2]],
                )
                self.predictions.append(prediction)
                self.final_times.append(final_times)
            # Pass two scores the stream's predictions with the usual metrics.
            self.replaying, self.client = self.config.steps > 0, -1
            metrics, rows = super().evaluate(**arguments)
        finally:
            self.replaying, self.streams, self.predictions = False, [], []
            model.load_state_dict(source)
        metrics.update({f"frozen_{key}": value for key, value in frozen.items()})
        metrics["adaptation_updates"] = float(self.updates)
        return metrics, rows
