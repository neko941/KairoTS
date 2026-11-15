"""TENT: fully test-time adaptation by entropy minimization (Wang et al., ICLR 2021).

Adapts only the affine parameters of the model's normalization layers, minimizing
the mean Shannon entropy of the class predictions on each unlabeled test batch.
Reference: https://github.com/DequanWang/tent (`tent.py`).
"""

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(kw_only=True)
class TENT:
    TASKS = ("Classification",)

    @dataclass(kw_only=True)
    class Config:
        name: str = "TENT"

    def __init__(self, *, config: Config) -> None:
        self.config = config

    def parameters(self, *, model: nn.Module) -> list[nn.Parameter]:
        """Normalization scales and shifts, the only parameters TENT updates."""
        # LayerNorm/GroupNorm only; BatchNorm also needs TENT's test-batch
        # statistics (train mode, no running stats) once a model uses it.
        found = [
            parameter
            for module in model.modules()
            if isinstance(module, nn.LayerNorm | nn.GroupNorm)
            for parameter in module.parameters(recurse=False)
        ]
        if not found:
            raise ValueError("TENT needs a model with LayerNorm or GroupNorm layers")
        return found

    def objective(self, *, prediction: Tensor) -> Tensor:
        """Mean entropy of the softmax over class scores `[batch, num_classes]`."""
        return (
            -(prediction.softmax(dim=1) * prediction.log_softmax(dim=1)).sum(1).mean()
        )
