"""TimesNet classification head: GELU, dropout, flatten and class projection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn
from torch.nn import functional

from kairots.models.TimesNet.Body import Body

if TYPE_CHECKING:
    from kairots.models.TimesNet import TimesNet


class Classification(nn.Module):
    def __init__(self, *, config: TimesNet.Config) -> None:
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(
            config.d_model * config.input_length, config.num_classes
        )

    def forward(self, *, batch_x: Tensor, body: Body) -> Tensor:
        activated = self.dropout(functional.gelu(body(batch_x=batch_x)))
        return self.projection(activated.flatten(start_dim=1))
