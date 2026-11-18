# ruff: noqa
import torch
from torch import nn
from torch.nn import functional as F

from kairots.models.iReformer.LSHAttention import LSHAttention, TOKEN_SELF_ATTN_VALUE


class FullQKAttention(nn.Module):
    def __init__(self, causal=False, dropout=0.0):
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, qk, v, query_len=None, input_mask=None, input_attn_mask=None, **kwargs
    ):
        b, seq_len, dim = qk.shape
        query_len = LSHAttention.default(query_len, seq_len)
        t = query_len

        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type_as(q)

        dot = torch.einsum("bie,bje->bij", q, qk) * (dim**-0.5)

        # qk attention requires tokens not attend to self
        i = torch.arange(t)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = LSHAttention.max_neg_value(dot)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, None] * input_mask[:, None, :]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            dot.masked_fill_(~mask, masked_value)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(
                input_attn_mask, (0, seq_len - input_attn_mask.shape[-1]), value=True
            )
            dot.masked_fill_(~input_attn_mask, masked_value)

        if self.causal:
            i, j = torch.triu_indices(t, t, 1)
            dot[:, i, j] = masked_value

        dot = dot.softmax(dim=-1)
        dot = self.dropout(dot)

        out = torch.einsum("bij,bje->bie", dot, v)

        return out, dot, torch.empty(0)
