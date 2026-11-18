# iInformer forecasting

This component is the paper's inverted Informer variant for multivariate forecasting.

## Sources

- Liu et al., [iTransformer](https://arxiv.org/abs/2310.06625), Section 5, full framework generality results.
- [Pinned author model](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/model/iInformer.py).
- [Pinned author attention code](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/layers/SelfAttention_Family.py).

## Paper-to-code map

| Paper operation | KairoTS implementation | Shape or convention | Check |
|---|---|---|---|
| Invert feature dimensions to create one token per variate | `layers/InvertedEmbedding.py` | Shared value projection with no positional embedding | Packed input forward test |
| Apply noncausal ProbSparse attention over variate tokens | `layers/Attention.py` | Reuses Informer's sampled-query attention with the author factor default of one | Author-reference parity test |
| Apply Transformer residual, feed-forward, and block normalization | `layers/InvertedEncoder.py` | Same post-normalized block as the paper | Author-reference parity test |
| Apply the variant's final encoder LayerNorm | `iInformer._encoder_norm` | Present in the pinned variant code | Variant parity test |

## Source choices

The factor default of one comes from the pinned author runner and is used because the iTransformer paper does not specify it for this variant.

The paper's core iTransformer algorithm omits a final encoder LayerNorm, but the pinned variant applies one; KairoTS preserves the variant code here.
