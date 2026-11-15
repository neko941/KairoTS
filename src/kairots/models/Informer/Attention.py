"""Informer Eq3/4 and Vaswani et al. per-position head concatenation."""

import math

import torch
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        heads: int,
        sparse: bool,
        causal: bool = False,
        sampling_factor: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.sparse = sparse
        self.causal = causal
        self.sampling_factor = sampling_factor
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def sparse_context(
        self, *, queries: Tensor, keys: Tensor, values: Tensor
    ) -> Tensor:
        query_count, key_count = queries.size(2), keys.size(2)
        sample_count = min(
            key_count, self.sampling_factor * math.ceil(math.log(max(2, key_count)))
        )
        selected_count = min(
            query_count,
            self.sampling_factor * math.ceil(math.log(max(2, query_count))),
        )
        sampled = torch.randint(
            key_count, (query_count, sample_count), device=queries.device
        )
        sampled_scores = (queries.unsqueeze(-2) * keys[:, :, sampled, :]).sum(-1)
        maximum = sampled_scores.amax(-1)
        # Eq4 explicitly fills unsampled pairs with zero.
        if sample_count < key_count:
            maximum = maximum.clamp_min(0)
        sparsity = maximum - sampled_scores.sum(-1) / key_count
        selected = sparsity.topk(selected_count, dim=-1, sorted=False).indices
        selected_queries = queries.gather(
            2, selected.unsqueeze(-1).expand(-1, -1, -1, queries.size(-1))
        )
        scores = selected_queries @ keys.transpose(-2, -1) / math.sqrt(queries.size(-1))
        if self.causal:
            scores = scores.masked_fill(
                torch.arange(key_count, device=queries.device) > selected.unsqueeze(-1),
                -torch.inf,
            )
            context = values.cumsum(dim=2)
        else:
            context = values.mean(dim=2, keepdim=True).expand(-1, -1, query_count, -1)
        updates = scores.softmax(dim=-1) @ values
        return context.scatter(2, selected.unsqueeze(-1).expand_as(updates), updates)

    def forward(self, *, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        batch, length, width = queries.shape
        projected_queries = (
            self.query_projection(queries)
            .reshape(batch, length, self.heads, width // self.heads)
            .transpose(1, 2)
        )
        projected_keys = (
            self.key_projection(keys)
            .reshape(batch, keys.size(1), self.heads, width // self.heads)
            .transpose(1, 2)
        )
        projected_values = (
            self.value_projection(values)
            .reshape(batch, values.size(1), self.heads, width // self.heads)
            .transpose(1, 2)
        )
        if self.sparse:
            context = self.sparse_context(
                queries=projected_queries, keys=projected_keys, values=projected_values
            )
        else:
            scores = projected_queries @ projected_keys.transpose(-2, -1)
            weights = self.dropout(
                (scores / math.sqrt(width // self.heads)).softmax(-1)
            )
            context = weights @ projected_values
        return self.output_projection(
            context.transpose(1, 2).reshape(batch, length, width)
        )
