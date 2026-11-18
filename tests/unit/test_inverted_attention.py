from __future__ import annotations

import math

import pytest
import torch


def test_flash_attention_matches_full_attention() -> None:
    from kairots.layers.FlashAttention import FlashAttention

    torch.manual_seed(3)
    module = FlashAttention(d_model=8, heads=2).eval()
    queries = torch.randn(2, 5, 8)
    keys = torch.randn(2, 7, 8)
    values = torch.randn(2, 7, 8)

    def split_heads(*, projected: torch.Tensor) -> torch.Tensor:
        return projected.reshape(2, -1, 2, 4).transpose(1, 2)

    query = split_heads(projected=module.query_projection(queries))
    key = split_heads(projected=module.key_projection(keys))
    value = split_heads(projected=module.value_projection(values))
    weights = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(4), dim=-1)
    expected = module.output_projection(
        (weights @ value).transpose(1, 2).reshape(2, 5, 8)
    )
    torch.testing.assert_close(
        module(queries=queries, keys=keys, values=values), expected
    )


def test_flow_attention_matches_author_equations() -> None:
    from kairots.layers.FlowAttention import FlowAttention

    torch.manual_seed(5)
    module = FlowAttention(d_model=8, heads=2).eval()
    features = torch.randn(2, 5, 8)
    query, key, value = (
        layer(features).reshape(2, 5, 2, 4).transpose(1, 2)
        for layer in (
            module.query_projection,
            module.key_projection,
            module.value_projection,
        )
    )
    query, key = query.sigmoid(), key.sigmoid()
    row = 1 / torch.einsum("bhld,bhd->bhl", query + 1e-6, key.sum(2) + 1e-6)
    column = 1 / torch.einsum("bhld,bhd->bhl", key + 1e-6, query.sum(2) + 1e-6)
    row_refinement = torch.einsum(
        "bhld,bhd->bhl", query + 1e-6, (key * column.unsqueeze(-1)).sum(2) + 1e-6
    )
    column_refinement = torch.einsum(
        "bhld,bhd->bhl", key + 1e-6, (query * row.unsqueeze(-1)).sum(2) + 1e-6
    )
    row_refinement = torch.sigmoid(row_refinement)
    column_refinement = torch.softmax(column_refinement, dim=-1) * key.size(2)
    key_values = key.transpose(-2, -1) @ (value * column_refinement.unsqueeze(-1))
    context = (query @ key_values) * row.unsqueeze(-1) * row_refinement.unsqueeze(-1)
    expected = module.output_projection(context.transpose(1, 2).reshape(2, 5, 8))
    torch.testing.assert_close(
        module(queries=features, keys=features, values=features), expected
    )


def test_reformer_attention_matches_padded_lsh_reference() -> None:
    reformer = pytest.importorskip("reformer_pytorch")
    from kairots.models.iReformer.Reformer import Reformer

    torch.manual_seed(7)
    module = Reformer(d_model=8, heads=2).eval()
    reference = reformer.LSHSelfAttention(
        dim=8, heads=2, bucket_size=4, n_hashes=4, causal=False
    ).eval()
    reference.load_state_dict(module.attention.state_dict())
    features = torch.randn(2, 7, 8)
    padded = torch.cat((features, features.new_zeros(2, 1, 8)), dim=1)
    torch.manual_seed(11)
    expected = reference(padded)[:, :7]
    torch.manual_seed(11)
    actual = module(queries=features, keys=features, values=features)
    torch.testing.assert_close(actual, expected)
