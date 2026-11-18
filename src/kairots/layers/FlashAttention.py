"""Exact full attention through PyTorch's flash-capable SDPA operator."""

from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention


class FlashAttention(nn.Module):
    def __init__(self, *, d_model: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, *, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        batch_size, query_length, width = queries.shape

        def split_heads(*, projected: Tensor) -> Tensor:
            return projected.reshape(
                batch_size, -1, self.heads, width // self.heads
            ).transpose(1, 2)

        context = scaled_dot_product_attention(
            split_heads(projected=self.query_projection(queries)),
            split_heads(projected=self.key_projection(keys)),
            split_heads(projected=self.value_projection(values)),
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output_projection(
            context.transpose(1, 2).reshape(batch_size, query_length, width)
        )
