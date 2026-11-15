"""Bottom-up reconciliation of hierarchical forecasts (Hyndman et al., 2011).

The model forecasts every series of the hierarchy; only the bottom-level forecasts
are kept. They are taken to raw units with each client's train-fitted scaler,
summed through the source's summing matrix into every series, and scaled back, so
the forecasts of every level add up exactly in raw units. The scores then follow
`Frozen`. Needs a `Hierarchical` source and each client's scaler (`inverse`).
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.data.sources.Hierarchical import Hierarchical
from kairots.evaluation.Frozen import Frozen
from kairots.registry import get


def bottom_up(
    *, prediction: Tensor, mean: Tensor, scale: Tensor, matrix: Tensor, bottom: Tensor
) -> Tensor:
    """Replace each series' scaled forecast by the sum of its bottom members'.

    `matrix` is [series, bottom] 0/1 and `bottom` the bottom series' channel indices.
    """
    raw = prediction * scale + mean
    return (raw[..., bottom] @ matrix.T - mean) / scale


@dataclass(kw_only=True)
class BottomUp(Frozen):
    TASKS = ("Forecasting",)

    @dataclass(kw_only=True)
    class Config(Frozen.Config):
        name: str = "BottomUp"
        # Reconciliation sums raw values, so the test phase is scored in raw units.
        inverse: bool = True

    def __init__(self, *, config: Config, loss_function: Any) -> None:
        super().__init__(config=config, loss_function=loss_function)
        self.matrix = torch.empty(0)
        self.bottom = torch.empty(0, dtype=torch.long)
        self.scales: list[tuple[Tensor, Tensor]] = []
        self.client = -1

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
        if scales is None:
            raise ValueError("BottomUp needs each client's scaler; keep inverse on")
        if not self.matrix.numel():
            source = experiment.data.source
            loaded = get(category="data/sources", name=source.name).load(config=source)
            if not isinstance(loaded, Hierarchical):
                raise ValueError(
                    f"BottomUp needs a Hierarchical source, not {source.name!r}"
                )
            fields = loaded.manifest_fields
            channels = list(source.input_channels)
            bottom = fields["bottom"]
            self.matrix = torch.tensor(
                [
                    [float(member in fields["summing"][name]) for member in bottom]
                    for name in channels
                ]
            )
            self.bottom = torch.tensor([channels.index(name) for name in bottom])
        self.scales, self.client = scales, -1
        return super().evaluate(
            phase=phase,
            model=model,
            clients=clients,
            parameters=parameters,
            scales=scales,
            experiment=experiment,
            view=view,
            task=task,
            device=device,
        )

    def start_client(self, *, phase: str, model: nn.Module) -> None:
        self.client += 1

    def predict(
        self,
        *,
        phase: str,
        model: nn.Module,
        batch_x: Tensor,
        windows: list[tuple[int, int]],
    ) -> Tensor:
        prediction = model(batch_x=batch_x)
        mean, scale = (
            value.to(prediction.device) for value in self.scales[self.client]
        )
        return bottom_up(
            prediction=prediction,
            mean=mean,
            scale=scale,
            matrix=self.matrix.to(prediction.device),
            bottom=self.bottom.to(prediction.device),
        )
