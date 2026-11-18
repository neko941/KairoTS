# DLinear forecasting

This page documents the ETTh1 forecasting reference.
DLinear's Gaussian and quantile heads are framework extensions and are not covered by this paper result.

## Sources

- Zeng et al., [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504), arXiv v3, Section 3.1 and Appendix.
- Original implementation: [`cure-lab/LTSF-Linear` at `0c113668`](https://github.com/cure-lab/LTSF-Linear/tree/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6).

The paper gives the temporal linear map as \(\hat X_i = W X_i\), then describes DLinear as the sum of linear projections of a trend component and its seasonal remainder.
The DLinear decomposition and summation are described in Section 3.1; the paper does not number these DLinear operations as separate equations.

## Paper-to-code map

| Source operation | KairoTS implementation | Shape / convention | Check |
|---|---|---|---|
| Moving-average trend and residual seasonal component (Section 3.1) | `models/DLinear/SeriesDecomposition.py` | Input `[batch, input_length, channels]`; trend uses edge replication padding so output length is preserved; seasonal is input minus trend | `test_dlinear_heads_share_the_body_and_match_reference` |
| Separate temporal linear projections, summed (Section 3.1) | `models/DLinear/Body.py` | Transpose to `[batch, channels, input_length]` for `nn.Linear`; return `[batch, output_length, channels]` | Same fixed-input reference test compares the shared seasonal/trend computation |
| Shared temporal weights across variates (Section 3.1) | `DLinear.Config.individual=False` by default | One seasonal and one trend map are reused across channels; `individual=True` is an explicit extension | The test checks shape for `individual=True`; its numerical reference comparison covers the shared paper setting |
| Forecast horizon selection | `models/DLinear/Forecasting.py` | The forecasting head keeps the last `output_length` body steps | Model output shape assertion in the same test |

## Initialization and protocol

The paper's Appendix uses uniform \(1/L\) weights for its **weight visualization**.
That is not the default training initialization.
The original model and KairoTS use the native `nn.Linear` initialization unless an explicit initialization policy is selected.
The fixed-input test checks that the default weights are not the visualization weights.

`configs/forecasting/ETTh1-DLinear.json` uses the paper's DLinear window and optimization settings: ETTh1, seven channels, 336 history steps, 96 forecast steps, moving-average kernel 25, shared weights, batch size 32, Adam at 0.005, ten training epochs, and validation-MSE checkpoint selection.
Standardization uses the first 8,640 training rows; the released 12/4/4-month cuts and drop-last training batches are explicit in the config.
The paper reports ETTh1 horizon-96 MSE 0.375 and MAE 0.399.
These metrics are scored on the configured scaled values.

This reference is for DLinear forecasting only.
A shape smoke test alone is not evidence of reproducing the paper's table; the paper-result record is maintained in the verification handoff.
