"""Transfer to a target population: each target client fine-tunes a model
pretrained elsewhere on its own data, after the style of FedHealth (Chen et al.,
"FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare",
IEEE Intelligent Systems, 2020).

Start from the source run with `initialization.name = "Pretrained"`; the split's
`train_windows` sets the few-shot budget (the last windows before the cut). Each
target client trains alone and keeps its model, as in `LocalOnly`, with every
parameter whose name matches a `frozen` pattern held at its pretrained value.
FedHealth's CORAL alignment of source and target features is not implemented.
"""

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

from torch import Tensor

from kairots.strategies.base import Strategy
from kairots.strategies.LocalOnly import LocalOnly


class Transfer(LocalOnly):
    @dataclass(kw_only=True)
    class Config(LocalOnly.Config):
        name: str = "Transfer"
        # fnmatch patterns of parameter names kept at their pretrained values.
        frozen: tuple[str, ...] = ()

    @dataclass(kw_only=True)
    class Client(Strategy.Client):
        frozen: tuple[str, ...] = field(default_factory=tuple)

        def fit(
            self,
            *,
            learner: Any,
            model: Any,
            optimizer: Any,
            batches: Any,
            epochs: int,
            device: Any,
            loss_function: Any,
            reference_parameters: dict[str, Tensor],
        ) -> float:
            """Freeze matched parameters (no gradient, so the optimizer skips them)."""
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(
                    not any(fnmatch(name, pattern) for pattern in self.frozen)
                )
            return super().fit(
                learner=learner,
                model=model,
                optimizer=optimizer,
                batches=batches,
                epochs=epochs,
                device=device,
                loss_function=loss_function,
                reference_parameters=reference_parameters,
            )

    def __init__(self, *, config: Config) -> None:
        self.config: Transfer.Config = config

    def client_update(self, *, worker: Any, job: Any) -> Any:
        return worker.update(
            job=job, client=self.Client(frozen=tuple(self.config.frozen))
        )
