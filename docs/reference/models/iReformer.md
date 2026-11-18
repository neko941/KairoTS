# iReformer forecasting

This component is the paper's inverted Reformer variant for multivariate forecasting.

## Sources

- Liu et al., [iTransformer](https://arxiv.org/abs/2310.06625), Section 5, full framework generality results.
- [Pinned author model](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/model/iReformer.py).
- The attention implementation is vendored from [reformer-pytorch 1.4.4](https://github.com/lucidrains/reformer-pytorch/tree/master/reformer_pytorch) under its MIT license.

## Paper-to-code map

| Paper operation | KairoTS implementation | Shape or convention | Check |
|---|---|---|---|
| Invert feature dimensions to create one token per variate | `layers/InvertedEmbedding.py` | Shared value projection with no positional embedding | Packed input forward test |
| Apply noncausal LSH self-attention over variate tokens | `models/iReformer/LSHAttention.py` | Author settings: bucket size four and four hashes | Author-reference parity test |
| Pad token count to a valid LSH bucket length | `Reformer.forward` | Zero-pad to a multiple of twice the bucket size and crop afterward | Odd token-count test |
| Apply Transformer residual, feed-forward, and block normalization | `layers/InvertedEncoder.py` | Same post-normalized block as the paper | Author-reference parity test |
| Apply the variant's final encoder LayerNorm | `iReformer._encoder_norm` | Present in the pinned variant code | Variant parity test |

## Source choices

The paper's core iTransformer algorithm omits a final encoder LayerNorm, but the pinned variant applies one; KairoTS preserves the variant code here.

The Reformer LSH implementation and its helper methods live in `models/iReformer/LSHAttention.py` with the required MIT notice, so no optional dependency is required.
