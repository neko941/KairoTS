"""Test-time adaptation on the test population, with a frozen comparator.

Train and validation phases are scored frozen. On the test phase the protocol first
scores the frozen model (reported as `frozen_*`), then adapts a copy of each
client's starting model on that client's test batches with the `adaptation`
method's unlabeled objective and scores the adapted predictions. Each batch is
predicted by the forward pass of its last update step (TENT's forward-and-adapt),
so `steps=0` reproduces the frozen scores. The method sees inputs only; targets
never reach an update. `reset="client"` carries the adapted state along one
client's test stream (TENT's online setting); `reset="batch"` restarts every batch
(episodic). Adapted state and optimizer moments never leave the test phase.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.evaluation.Frozen import Frozen
from kairots.learners.base import Learning
from kairots.optimizers.base import UpdateBudget
from kairots.registry import get


@dataclass(kw_only=True)
class Adapted(Frozen):
    @dataclass(kw_only=True)
    class Config(Frozen.Config, Learning.Config, UpdateBudget.Config):
        name: str = "Adapted"
        adaptation: str = "TENT"
        # TENT's defaults: one Adam step per test batch at learning rate 1e-3.
        learning_rate: float = 1e-3
        reset: str = "client"

        def __post_init__(self) -> None:
            if self.reset not in ("client", "batch"):
                raise ValueError("reset must be 'client' or 'batch'")
            if self.steps < 0:
                raise ValueError("steps must be non-negative")

    def __init__(self, *, config: Config, loss_function: Any) -> None:
        super().__init__(config=config, loss_function=loss_function)
        self.config: Adapted.Config = config
        method_type = get(category="adaptation", name=config.adaptation)
        self.method = method_type(config=method_type.Config())
        self.adapting = False
        self.updates = 0
        self.client_state: dict[str, Tensor] = {}
        self.optimizer: torch.optim.Optimizer | None = None
        self.adapted: list[nn.Parameter] = []

    def start_client(self, *, phase: str, model: nn.Module) -> None:
        if self.adapting:
            self.client_state = {
                key: value.clone() for key, value in model.state_dict().items()
            }
            self._restart(model=model)

    def _restart(self, *, model: nn.Module) -> None:
        model.load_state_dict(self.client_state)
        self.adapted = self.method.parameters(model=model)
        optimizer_type = get(category="optimizers", name=self.config.optimizer.name)
        self.optimizer = optimizer_type(config=self.config.optimizer).build(
            parameters=self.adapted,
            learning_rate=self.config.learning_rate,
            fused=False,
        )

    def predict(
        self,
        *,
        phase: str,
        model: nn.Module,
        batch_x: Tensor,
        windows: list[tuple[int, int]],
    ) -> Tensor:
        if not self.adapting or self.config.steps == 0:
            return super().predict(
                phase=phase, model=model, batch_x=batch_x, windows=windows
            )
        if self.config.reset == "batch":
            self._restart(model=model)
        assert self.optimizer is not None
        prediction = batch_x
        with torch.enable_grad():
            for _ in range(self.config.steps):
                prediction = model(batch_x=batch_x)
                loss = self.method.objective(prediction=prediction)
                gradients = torch.autograd.grad(loss, self.adapted)
                for parameter, gradient in zip(self.adapted, gradients, strict=True):
                    parameter.grad = gradient
                self.optimizer.step()
                self.updates += 1
        return prediction.detach()

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
        source = {key: value.clone() for key, value in model.state_dict().items()}
        # Every client starts from its own parameters, or from the shared model.
        arguments["parameters"] = parameters or [source] * len(clients)
        self.adapting, self.updates = True, 0
        try:
            metrics, rows = super().evaluate(**arguments)
        finally:
            self.adapting, self.optimizer, self.adapted = False, None, []
            model.load_state_dict(source)
        metrics.update({f"frozen_{key}": value for key, value in frozen.items()})
        metrics["adaptation_updates"] = float(self.updates)
        return metrics, rows
