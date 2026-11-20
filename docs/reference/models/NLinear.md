# NLinear forecasting

This page documents the ETTh1 forecasting reference.
NLinear's `individual` option is the official alternative to shared weights; it is not a framework extension.

## Sources

- Zeng et al., [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504), arXiv v3, Section 3.1 and Appendix.
- Original implementation: [`cure-lab/LTSF-Linear` at `0c113668`](https://github.com/cure-lab/LTSF-Linear/tree/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6).

The paper describes NLinear as subtracting the last value of the input sequence, applying a linear layer, and adding the subtracted value back.
The subtraction is a simple normalization for a distribution shift; it is not a separate trainable step.

## Paper-to-code map

| Source operation | KairoTS implementation | Shape / convention | Check |
|---|---|---|---|
| Subtract the last history step before the linear map (Section 3.1) | `models/NLinear/Forecasting.py` | Input `[batch, input_length, channels]`; the last step `[:, -1:, :]` is detached and subtracted | `test_nlinear_subtracts_and_restores_the_last_history_step` |
| Temporal linear projection | `models/NLinear/Forecasting.py` | Transpose to `[batch, channels, input_length]` for `nn.Linear`; return `[batch, output_length, channels]` | The same fixed-input test compares the full forward pass with the reference math |
| Add the subtracted step back | `models/NLinear/Forecasting.py` | The forecast plus the last history step, broadcast over the horizon | The test shifts the whole history window and checks the forecast shifts with it |
| Shared temporal weights across variates (Section 3.1) | `NLinear.Config.individual=False` by default | One map is reused across channels; `individual=True` uses one map per variate, as the official option | The same test checks both settings |
| Forecast horizon selection | `models/NLinear/Forecasting.py` | The linear map outputs exactly `output_length` steps | Model output shape assertion in the same test |

## Initialization and protocol

The original implementation and KairoTS use the native `nn.Linear` initialization unless an explicit initialization policy is selected.

`configs/zeng2023-dlinear/Table2-NLinear.json` sweeps the nine multivariate benchmarks of the paper's Table 2.
The shared settings are 336 history steps (104 for ILI), the released ETT cuts or the paper's 70/10/20 chronological split, shared weights, Adam, validation-MSE checkpoint selection, and ten one-epoch rounds with early stopping.
Each run overrides its dataset, horizon, batch size and learning rate; ILI uses horizons 24, 36, 48 and 60, the others 96, 192, 336 and 720.
The paper reports ETTh1 horizon-96 MSE 0.374 and MAE 0.394.
These metrics are scored on the configured scaled values.

This reference is for NLinear forecasting only.
A shape smoke test alone is not evidence of reproducing the paper's table; the paper-result record is maintained in the verification handoff.
