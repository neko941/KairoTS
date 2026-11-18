"""Shared history projection for inverted variate and optional calendar tokens."""

import torch
from torch import Tensor, nn


class InvertedEmbedding(nn.Module):
    def __init__(self, *, input_length: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(input_length, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *, features: Tensor, marks: Tensor | None = None) -> Tensor:
        tokens = features.permute(0, 2, 1)
        if marks is not None:
            tokens = torch.cat((tokens, marks.permute(0, 2, 1)), dim=1)
        return self.dropout(self.value_embedding(tokens))
