"""Flow attention from the iTransformer Flowformer ablation."""

import torch
from torch import Tensor, nn


class FlowAttention(nn.Module):
    def __init__(self, *, d_model: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, *, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        batch_size, query_length, width = queries.shape
        key_length = keys.size(1)
        head_width = width // self.heads

        def split_heads(*, projected: Tensor) -> Tensor:
            return projected.reshape(batch_size, -1, self.heads, head_width).transpose(
                1, 2
            )

        query = torch.sigmoid(split_heads(projected=self.query_projection(queries)))
        key = torch.sigmoid(split_heads(projected=self.key_projection(keys)))
        value = split_heads(projected=self.value_projection(values))
        row_normalizer = 1 / torch.einsum(
            "bhld,bhd->bhl", query + 1e-6, key.sum(dim=2) + 1e-6
        )
        column_normalizer = 1 / torch.einsum(
            "bhld,bhd->bhl", key + 1e-6, query.sum(dim=2) + 1e-6
        )
        row_refinement = torch.einsum(
            "bhld,bhd->bhl",
            query + 1e-6,
            (key * column_normalizer.unsqueeze(-1)).sum(dim=2) + 1e-6,
        )
        column_refinement = torch.einsum(
            "bhld,bhd->bhl",
            key + 1e-6,
            (query * row_normalizer.unsqueeze(-1)).sum(dim=2) + 1e-6,
        )
        row_refinement = torch.sigmoid(row_refinement * (query_length / key_length))
        column_refinement = torch.softmax(column_refinement, dim=-1) * key_length
        key_values = key.transpose(-2, -1) @ (value * column_refinement.unsqueeze(-1))
        context = (
            (query @ key_values)
            * row_normalizer.unsqueeze(-1)
            * row_refinement.unsqueeze(-1)
        )
        return self.output_projection(
            context.transpose(1, 2).reshape(batch_size, query_length, width)
        )
