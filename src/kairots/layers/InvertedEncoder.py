"""Noncausal full-attention and positionwise feed-forward encoder block."""

from typing import Literal

import torch
from torch import Tensor, nn


class InvertedEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        attention_heads: int,
        dropout: float,
        activation: Literal["gelu", "relu"],
        attention: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.attention_heads = attention_heads
        self.attention = attention
        self.query_projection = (
            nn.Linear(d_model, d_model) if attention is None else None
        )
        self.key_projection = nn.Linear(d_model, d_model) if attention is None else None
        self.value_projection = (
            nn.Linear(d_model, d_model) if attention is None else None
        )
        self.out_projection = nn.Linear(d_model, d_model) if attention is None else None
        self.attention_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = (
            torch.nn.functional.gelu
            if activation == "gelu"
            else torch.nn.functional.relu
        )

    def forward(self, *, features: Tensor) -> Tensor:
        if self.attention is not None:
            attended = self.attention(
                queries=features,
                keys=features,
                values=features,
            )
        else:
            batch_size, token_count, model_width = features.shape
            head_width = model_width // self.attention_heads

            def split_heads(*, projected: Tensor) -> Tensor:
                return projected.reshape(
                    batch_size, token_count, self.attention_heads, head_width
                ).transpose(1, 2)

            assert self.query_projection is not None
            assert self.key_projection is not None
            assert self.value_projection is not None
            assert self.out_projection is not None
            query = split_heads(projected=self.query_projection(features))
            key = split_heads(projected=self.key_projection(features))
            value = split_heads(projected=self.value_projection(features))
            weights = torch.softmax(
                query @ key.transpose(-2, -1) / (head_width**0.5), dim=-1
            )
            attended = self.attention_dropout(weights) @ value
            attended = attended.transpose(1, 2).reshape(
                batch_size, token_count, model_width
            )
            attended = self.out_projection(attended)
        hidden = self.norm1(features + self.dropout(attended))
        projected = self.conv1(hidden.transpose(1, 2))
        projected = self.dropout(self.activation(projected))
        projected = self.dropout(self.conv2(projected).transpose(1, 2))
        return self.norm2(hidden + projected)
