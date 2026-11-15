"""Non-private one-shot ridge fusion, Alsulaimawi (2026), arXiv:2601.08216v1.

Algorithm 1 and the shared random projection extension. Each forecast horizon is
an independent ridge problem; channels are observations in one aligned feature
space. The intercept is fixed at zero, as in the paper's quadratic objective, so
the broadcast model is the weight alone. Clients send the upper triangle of the
Gram and the moment (Theorem 4); the server solves by Cholesky (Remark 5). Every
value is float32, on the wire and in the solve.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from kairots.runtime.contracts import ClientState, Job, Message, Result
from kairots.strategies.Traditional import Traditional


class FedRidge(Traditional):
    TASKS = ("Forecasting",)
    MODELS = ("Linear",)
    REQUIRES = {
        "task.name": ("Forecasting",),
        "groups": ((),),
        "client_model": (None,),
        "model.name": ("Linear",),
        "learner.name": ("Stateless",),
        "learner.epochs": (0,),
        "training.name": ("Offline",),
        "data.view.name": ("Offline",),
        "data.augmentation": (None,),
        "task.calendar": ((),),
        "task.target_channels": ((),),
        "evaluation.name": ("Frozen",),
        "evaluation.finetune_epochs": (0,),
    }

    @dataclass(kw_only=True)
    class Config(Traditional.Config):
        name: str = "FedRidge"
        iterations: int = 1
        gamma: float = 0.1
        projection_seed: int = 20260922

        def __post_init__(self) -> None:
            if self.iterations != 1:
                raise ValueError("FedRidge requires exactly one iteration")
            if not math.isfinite(self.gamma) or self.gamma <= 0:
                raise ValueError("FedRidge gamma must be positive and finite")

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def projection(self, *, input_length: int) -> Tensor | None:
        if input_length <= 1000:
            return None
        dimension = int(0.4 * input_length)
        return torch.randn(
            input_length,
            dimension,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(self.config.projection_seed),
        ) / math.sqrt(dimension)

    def build_downlink(
        self, *, global_params: dict[str, Tensor], client_id: int
    ) -> Message:
        # Sufficient statistics are independent of the initial model.
        return Message(sender="server", receiver=f"client:{client_id}")

    def client_update(self, *, worker: Any, job: Job) -> Result:
        if job.operation != "fit" or job.operation_id[0] != 0:
            raise ValueError("FedRidge accepts only the first fitting round")
        experiment = worker.client_experiment(client_id=job.client_id)
        input_length, output_length, offset_length = experiment.lengths()
        data = worker.client_data(client_id=job.client_id)
        _, view, _ = worker.components(client_id=job.client_id)
        projection = self.projection(input_length=input_length)
        dimension = input_length if projection is None else projection.shape[1]
        gram = torch.zeros(dimension, dimension, dtype=torch.float32)
        moment = torch.zeros(dimension, output_length, dtype=torch.float32)
        windows = data.splits["train"]
        if not windows:
            raise ValueError("FedRidge requires nonempty training windows")
        for batch_x, batch_y in view.batches(
            values=data.values,
            windows=windows,
            input_length=input_length,
            output_length=output_length,
            offset_length=offset_length,
            batch_size=experiment.data.source.batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(job.seed),
        ):
            if (
                batch_x.ndim != 3
                or batch_y.ndim != 3
                or batch_x.shape[0] != batch_y.shape[0]
                or batch_x.shape[2] != batch_y.shape[2]
                or batch_x.shape[1] != input_length
                or batch_y.shape[1] != output_length
            ):
                raise ValueError("FedRidge needs aligned [batch, steps, channels]")
            features = batch_x.transpose(1, 2).reshape(-1, input_length).float().cpu()
            target_values = (
                batch_y.transpose(1, 2).reshape(-1, output_length).float().cpu()
            )
            if (
                not torch.isfinite(features).all()
                or not torch.isfinite(target_values).all()
            ):
                raise ValueError("FedRidge requires finite features and targets")
            if projection is not None:
                features = features @ projection
            gram += features.T @ features
            moment += features.T @ target_values
        triangle = torch.triu_indices(dimension, dimension)
        return Result(
            operation_id=job.operation_id,
            uplink=Message(
                sender=job.downlink.receiver,
                receiver=job.downlink.sender,
                tensors={"gram": gram[triangle[0], triangle[1]], "moment": moment},
            ),
            state=ClientState(version=job.state.version + 1),
            metrics={},
            train_samples=len(windows),
        )

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        # Clients reject empty training windows; the server sees only the statistics.
        if not uplinks:
            raise ValueError("FedRidge requires nonempty client uplinks")
        input_length = global_params["forecasting.linear.weight"].shape[1]
        output_length = global_params["forecasting.linear.weight"].shape[0]
        projection = self.projection(input_length=input_length)
        dimension = input_length if projection is None else projection.shape[1]
        packed_size = dimension * (dimension + 1) // 2
        gram = torch.zeros(dimension, dimension, dtype=torch.float32)
        moment = torch.zeros(dimension, output_length, dtype=torch.float32)
        triangle = torch.triu_indices(dimension, dimension)
        for message in uplinks:
            tensors = message.tensors
            if (
                set(tensors) != {"gram", "moment"}
                or tensors["gram"].shape != (packed_size,)
                or tensors["moment"].shape != moment.shape
                or any(not torch.isfinite(value).all() for value in tensors.values())
            ):
                raise ValueError("invalid FedRidge sufficient statistics")
            gram[triangle[0], triangle[1]] += tensors["gram"].float().cpu()
            moment += tensors["moment"].float().cpu()
        gram = gram + gram.T - torch.diag(gram.diag())
        gram += self.config.gamma * torch.eye(dimension, dtype=torch.float32)
        weights = torch.cholesky_solve(moment, torch.linalg.cholesky(gram))
        if projection is not None:
            weights = projection @ weights
        return {
            "forecasting.linear.weight": weights.T.contiguous(),
            "forecasting.linear.bias": torch.zeros(output_length, dtype=torch.float32),
        }

    def broadcast(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> list[Message]:
        # The fixed zero intercept is known locally and does not go on the wire.
        return [
            Message(
                sender="server",
                receiver=message.sender,
                tensors={"weight": global_params["forecasting.linear.weight"]},
            )
            for message in uplinks
        ]
