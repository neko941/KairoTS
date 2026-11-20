"""_stack used by TiDE."""

from torch import nn

from kairots.models.TiDE.ResBlock import ResBlock


def _stack(
    *,
    count: int,
    input_dimension: int,
    hidden: int,
    output_dimension: int,
    dropout: float,
    layer_norm: bool,
) -> nn.ModuleList:
    """`count` blocks from input_dimension to output_dimension."""
    return nn.ModuleList(
        ResBlock(
            input_dimension=input_dimension if index == 0 else hidden,
            hidden=hidden,
            output_dimension=output_dimension if index == count - 1 else hidden,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        for index in range(count)
    )
