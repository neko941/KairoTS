"""TimeKAN's per-level embedding of a univariate sequence into `d_model` channels."""

from torch import Tensor, nn


class ValueEmbedding(nn.Module):
    """`[rows, length, 1]` to `[rows, length, d_model]`, then dropout."""

    def __init__(self, *, d_model: int, dropout: float, convolutional: bool) -> None:
        super().__init__()
        self.convolutional = convolutional
        # paper vs code: the paper embeds each level with a Linear layer (eq. 2); the
        # official code uses a kernel-3 circular Conv1d token embedding without bias
        # (`TokenEmbedding`, kaiming-normal fan_in, leaky_relu).
        if convolutional:
            self.layer: nn.Module = nn.Conv1d(
                1,
                d_model,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
                bias=False,
            )
            nn.init.kaiming_normal_(
                self.layer.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
        else:
            self.layer = nn.Linear(1, d_model)
        # official code: `DataEmbedding_wo_pos` applies dropout (paper silent).
        self.dropout = nn.Dropout(dropout)

    def forward(self, *, features: Tensor) -> Tensor:
        if self.convolutional:
            return self.dropout(self.layer(features.transpose(1, 2)).transpose(1, 2))
        return self.dropout(self.layer(features))
