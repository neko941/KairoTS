"""Reformer LSH attention used by the iTransformer Reformer ablation."""

import torch
from torch import Tensor, nn

from kairots.models.iReformer.LSHSelfAttention import LSHSelfAttention


class Reformer(nn.Module):
    def __init__(self, *, d_model: int, heads: int) -> None:
        super().__init__()
        self.bucket_size = 4
        self.attention = LSHSelfAttention(
            dim=d_model,
            heads=heads,
            bucket_size=self.bucket_size,
            n_hashes=4,
            causal=False,
        )

    def forward(self, *, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        sequence_length = queries.size(1)
        padding_length = (-sequence_length) % (self.bucket_size * 2)
        if padding_length:
            padding = queries.new_zeros(
                queries.size(0), padding_length, queries.size(2)
            )
            queries = torch.cat((queries, padding), dim=1)
        return self.attention(queries)[:, :sequence_length]
