# iFlowformer forecasting

This component is the paper's inverted Flowformer variant for multivariate forecasting.

## Sources

- Liu et al., [iTransformer](https://arxiv.org/abs/2310.06625), Section 5, full framework generality results.
- [Pinned author model](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/model/iFlowformer.py).
- [Pinned FlowAttention implementation](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/layers/SelfAttention_Family.py).

## Paper-to-code map

| Paper operation | KairoTS implementation | Shape or convention | Check |
|---|---|---|---|
| Invert feature dimensions to create one token per variate | `layers/InvertedEmbedding.py` | Shared value projection with no positional embedding | Packed input forward test |
| Apply FlowAttention over variate tokens | `layers/FlowAttention.py` | Sigmoid kernel, row and column normalization, refinement, competition, and allocation | Author-reference parity test |
| Apply Transformer residual, feed-forward, and block normalization | `layers/InvertedEncoder.py` | Same post-normalized block as the paper | Author-reference parity test |
| Apply the variant's final encoder LayerNorm | `iFlowformer._encoder_norm` | Present in the pinned variant code | Variant parity test |

## Source choices

The attention layer follows the pinned author's FlowAttention implementation, including its additive `1e-6` normalizers and lack of attention-probability dropout.

The paper's core iTransformer algorithm omits a final encoder LayerNorm, but the pinned variant applies one; KairoTS preserves the variant code here.
