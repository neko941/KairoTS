# ruff: noqa
from functools import partial

import torch
from torch import nn

from kairots.models.iReformer.FullQKAttention import FullQKAttention
from kairots.models.iReformer.LSHAttention import LSHAttention


class LSHSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        bucket_size=64,
        n_hashes=8,
        causal=False,
        dim_head=None,
        attn_chunks=1,
        random_rotations_per_head=False,
        attend_across_buckets=True,
        allow_duplicate_attention=True,
        num_mem_kv=0,
        one_value_head=False,
        use_full_attn=False,
        full_attn_thres=None,
        return_attn=False,
        post_attn_dropout=0.0,
        dropout=0.0,
        n_local_attn_heads=0,
        **kwargs,
    ):
        super().__init__()
        assert dim_head or (dim % heads) == 0, (
            "dimensions must be divisible by number of heads"
        )
        assert n_local_attn_heads < heads, (
            "local attention heads must be less than number of heads"
        )
        if n_local_attn_heads != 0:
            raise ValueError(
                "The KairoTS Reformer variant does not use local attention heads."
            )

        dim_head = LSHAttention.default(dim_head, dim // heads)
        dim_heads = dim_head * heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_chunks = LSHAttention.default(attn_chunks, 1)

        self.v_head_repeats = heads if one_value_head else 1
        v_dim = dim_heads // self.v_head_repeats

        self.toqk = nn.Linear(dim, dim_heads, bias=False)
        self.tov = nn.Linear(dim, v_dim, bias=False)
        self.to_out = nn.Linear(dim_heads, dim)

        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
            random_rotations_per_head=random_rotations_per_head,
            attend_across_buckets=attend_across_buckets,
            allow_duplicate_attention=allow_duplicate_attention,
            return_attn=return_attn,
            dropout=dropout,
            **kwargs,
        )
        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.use_full_attn = use_full_attn
        self.full_attn_thres = LSHAttention.default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = (
            nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True))
            if num_mem_kv > 0
            else None
        )

        self.callback = None

    def forward(
        self,
        x,
        keys=None,
        input_mask=None,
        input_attn_mask=None,
        context_mask=None,
        pos_emb=None,
        **kwargs,
    ):
        device, dtype = x.device, x.dtype
        b, t, e, h, m = *x.shape, self.heads, self.num_mem_kv

        mem_kv = LSHAttention.default(
            self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device)
        )
        mem = mem_kv.expand(b, m, -1)

        keys = LSHAttention.default(
            keys, torch.empty(b, 0, e, dtype=dtype, device=device)
        )
        c = keys.shape[1]

        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres

        x = torch.cat((x, mem, keys), dim=1)
        qk = self.toqk(x)
        v = self.tov(x)
        v = v.repeat(1, 1, self.v_head_repeats)

        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2)

        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

        merge_batch_and_heads = partial(LSHAttention.merge_dims, 0, 1)

        qk, v = map(merge_heads, (qk, v))
        qk, v = map(merge_batch_and_heads, (qk, v))
        lsh_h = h

        masks = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = LSHAttention.default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = LSHAttention.default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(LSHAttention.expand_dim(1, lsh_h, mask))
            masks["input_mask"] = mask

        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(
                LSHAttention.expand_dim(1, lsh_h, input_attn_mask)
            )
            masks["input_attn_mask"] = input_attn_mask

        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn
        partial_attn_fn = partial(attn_fn, query_len=t, pos_emb=pos_emb, **kwargs)
        attn_fn_in_chunks = LSHAttention.process_inputs_chunk(
            partial_attn_fn, chunks=self.attn_chunks
        )

        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)

        if self.callback is not None:
            self.callback(attn.reshape(b, lsh_h, t, -1), buckets.reshape(b, lsh_h, -1))

        out = split_heads(out).view(b, t, -1)
        out = self.to_out(out)
        return self.post_attn_dropout(out)
