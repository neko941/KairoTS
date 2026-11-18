# iTransformer forecasting

This page documents KairoTS's forecasting-only implementation of the iTransformer architecture.

## Sources

- Liu et al., [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625), ICLR 2024, arXiv v4.
- Official repository: [thuml/iTransformer at commit c2426e68](https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124).
- The paper/code discrepancy is documented in [iTransformer source notes](../../explanation/notes.md#itransformer-source-notes).

## Paper-to-code map

| Paper operation | KairoTS implementation | Shape or convention | Check |
|---|---|---|---|
| Embed every complete variate history with a shared linear projection | `models/iTransformer/DataEmbedding.py` | `[batch, channels, input_length]` becomes one token per channel, with no positional embedding | `test_itransformer_forecasts_packed_calendar_inputs_and_backpropagates` |
| Append optional time-feature tokens through the same projection | `DataEmbedding.forward` | Only historical calendar marks are passed, matching the original forecasting path | Same packed-calendar test |
| Apply full noncausal scaled dot-product attention over variate tokens | `models/iTransformer/EncoderLayer.py` | Head dimension is `d_model / attention_heads`; the attention mask is absent | Same forward and gradient test |
| Add the attention residual and LayerNorm, then the feed-forward residual and LayerNorm | `EncoderLayer.forward` | Post-normalization per block; no additional encoder-final LayerNorm, following the paper algorithm | Same forward test |
| Project each token to the forecast horizon with shared weights | `iTransformer.forward` | `[batch, variates, d_model]` maps to `[batch, output_length, target_channels]` | Same packed-calendar test |
| Generalize weights to a changed variate count | `iTransformer.Config`, `RevIN` | Channel count changes do not change parameter shapes | `test_itransformer_weights_are_independent_of_variate_count` |

## Scope

The original repository implements forecasting only; its TSLib multi-task heads are not part of this model.

The paper's optional variate-subsampling training procedure belongs to the training loop and is outside this forecasting-model component.

The model accepts the forecasting task's packed history and optional calendar layout; future calendar marks are not read.
