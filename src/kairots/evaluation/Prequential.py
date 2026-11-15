"""Prequential (test-then-train) scoring of the test stream, with a frozen comparator.

Train and validation phases are scored frozen. On the test phase the protocol first
scores the frozen model (reported as `frozen_*`), then walks each client's test
windows in time order: each batch is predicted and scored first, and the `regime`
(`training/`) learns from it only once its targets are observed. Each
client starts from its own parameters and a fresh learner optimizer; the model is
restored afterwards, so online updates never leave the test phase.

Retention: each client's model at the end of its test stream is scored on that
client's validation windows, the experience just before the stream, and reported
as `retained_*`; `forgetting_*` is that score minus the frozen model's (positive
means worse for an error metric).
"""

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

from kairots.evaluation.Frozen import Frozen
from kairots.registry import get
from kairots.training.Online import Online


@dataclass(kw_only=True)
class Prequential(Frozen):
    TASKS = ("Forecasting",)
    REQUIRES = {"evaluation.regime": ("Online",)}

    @dataclass(kw_only=True)
    class Config(Frozen.Config, Online.Config):
        name: str = "Prequential"
        regime: str = "Online"

    def __init__(self, *, config: Config, loss_function: Any) -> None:
        super().__init__(config=config, loss_function=loss_function)
        self.config: Prequential.Config = config
        regime_type = get(category="training", name=config.regime)
        # The regime takes the fields of this config that its own Config declares.
        options = {
            item.name: getattr(config, item.name)
            for item in fields(regime_type.Config)
            if item.name != "name" and hasattr(config, item.name)
        }
        self.regime = regime_type(config=regime_type.Config(**options))
        self.streaming = False
        self.final_states: list[dict[str, Tensor]] = []
        self.learner: Any = None
        # (input_length, output_length) of the stream's windows.
        self.lengths: tuple[int, int] = (0, 0)

    def start_client(self, *, phase: str, model: nn.Module) -> None:
        if self.streaming:
            self.regime.start(optimizer=self.learner.optimizer(model=model))

    def end_client(self, *, phase: str, model: nn.Module) -> None:
        if self.streaming:
            self.final_states.append(
                {key: value.clone() for key, value in model.state_dict().items()}
            )

    def predict(
        self,
        *,
        phase: str,
        model: nn.Module,
        batch_x: Tensor,
        windows: list[tuple[int, int]],
    ) -> Tensor:
        if self.streaming:
            # The batch is predicted at once, so its earliest origin bounds what the
            # stream has observed: history ends at start + input_length.
            observed = min(start for start, _ in windows) + self.lengths[0]
            self.regime.release(
                model=model, observed=observed, loss_function=self.loss_function
            )
        return super().predict(
            phase=phase, model=model, batch_x=batch_x, windows=windows
        )

    def observe(
        self,
        *,
        phase: str,
        batch_x: Tensor,
        batch_y: Tensor,
        windows: list[tuple[int, int]],
    ) -> None:
        if self.streaming:
            self.regime.observe(
                batch_x=batch_x,
                batch_y=batch_y,
                ends=[target + self.lengths[1] for _, target in windows],
            )

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
        frozen, rows = super().evaluate(**arguments)
        if phase != "test":
            return frozen, rows
        learner_type = get(category="learners", name=experiment.learner.name)
        self.learner = learner_type(config=experiment.learner)
        if not hasattr(self.learner, "optimizer"):
            raise ValueError("Prequential needs a gradient learner with an optimizer")
        input_length, output_length, _ = experiment.lengths()
        self.lengths = (input_length, output_length)
        source = {key: value.clone() for key, value in model.state_dict().items()}
        # Every client starts from its own parameters, or from the shared model.
        arguments["parameters"] = parameters or [source] * len(clients)
        self.streaming, self.regime.updates, self.final_states = True, 0, []
        try:
            metrics, rows = super().evaluate(**arguments)
            self.streaming = False
            before: dict[str, float] = {}
            after: dict[str, float] = {}
            # Forgetting needs validation windows; without them it is not reported.
            if any(client[1]["validation"] for client in clients):
                validation = {**arguments, "phase": "validation"}
                before, _ = super().evaluate(**validation)
                after, _ = super().evaluate(
                    **{**validation, "parameters": self.final_states}
                )
        finally:
            self.streaming = False
            model.load_state_dict(source)
        metrics.update({f"retained_{key}": value for key, value in after.items()})
        metrics.update({f"forgetting_{key}": after[key] - before[key] for key in after})
        metrics.update({f"frozen_{key}": value for key, value in frozen.items()})
        metrics["online_updates"] = float(self.regime.updates)
        return metrics, rows
