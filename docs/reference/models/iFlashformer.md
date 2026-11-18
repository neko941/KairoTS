# iFlashformer forecasting

This component is the paper's inverted Flashformer variant for multivariate forecasting.

## Sources

- Liu et al., [iTransformer](https://arxiv.org/abs/2310.06625), Section 5, full framework generality results.
- [Pinned author model](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/model/iFlashformer.py).
- [Pinned author attention code](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/layers/SelfAttention_Family.py).

## Paper-to-code map

| Paper operation | KairoTS implementation | Shape or convention | Check |
|---|---|---|---|
| Invert feature dimensions to create one token per variate | `layers/InvertedEmbedding.py` | Shared value projection with no positional embedding | Packed input forward test |
| Apply FlashAttention over variate tokens | `layers/FlashAttention.py` | PyTorch SDPA preserves exact full-attention semantics and uses its flash backend when supported | SDPA equivalence test |
| Apply Transformer residual, feed-forward, and block normalization | `layers/InvertedEncoder.py` | Same post-normalized block as the paper | Author-reference parity test |
| Apply the variant's final encoder LayerNorm | `iFlashformer._encoder_norm` | Present in the pinned variant code | Variant parity test |

## Source choices

The author implementation's FlashAttention routine hardcodes CUDA and does not apply its declared attention dropout.

KairoTS uses PyTorch SDPA with zero attention-probability dropout, matching that routine while allowing supported backends to select FlashAttention.

The paper's core iTransformer algorithm omits a final encoder LayerNorm, but the pinned variant applies one; KairoTS preserves the variant code here.
