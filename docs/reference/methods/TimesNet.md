# TimesNet

TimesNet uses dominant Fourier periods to reshape a time series into 2D views,
then applies multi-scale Inception blocks. This page maps the paper's period rule
and the implemented forecasting, classification, and imputation paths.

## Sources

- Wu et al., [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://arxiv.org/abs/2210.02186), ICLR 2023, arXiv v3, Sections 3.1–3.2 and Appendix A.
- Official implementation: [thuml/Time-Series-Library at `4e938a17`](https://github.com/thuml/Time-Series-Library/tree/4e938a1767106324dd753b2a44832bf870a0252e).

## Paper-to-code map

| Paper operation | KairoTS implementation | Shape or convention | Check |
|---|---|---|---|
| Select the top `k` non-DC Fourier frequencies | `models/TimesNet/TimesBlock.py` | FFT is over time; amplitudes average across batch and hidden channels | `test_timesnet_period_uses_paper_ceiling_for_nondivisible_frequency` |
| Set period `p = ceil(T / f)` and pad to a multiple of `p` before the 2D reshape (Eq. 1) | `TimesBlock.forward` | `[batch, time, hidden]` becomes `[batch, hidden, ceil(T/p), p]`; TSLib floors `T/f`, while KairoTS follows the paper | Same explicit nondivisible-period fixture |
| Apply shared multi-scale Inception convolutions, weight views by normalized frequency amplitudes, and add the residual (Eq. 2) | `models/TimesNet/InceptionBlock.py`, `TimesBlock.forward` | Every view returns to `[batch, time, hidden]` before weighted summation | TimesNet model tests |
| Embed values and positions, then stack TimesBlocks | `models/TimesNet/DataEmbedding.py`, `TokenEmbedding.py`, `Body.py`, `layers/PositionalEmbedding.py` | Forecasting and classification use `[batch, time, channels]` | `test_timesnet_task_outputs_match_tslib_contract` |
| Forecasting and reconstruction heads | `models/TimesNet/Forecasting.py` | Forecasting extends the time axis then selects the last `output_length`; reconstruction preserves the input length | Same task-output contract test |
| Classification head | `models/TimesNet/Classification.py` | GELU, dropout, flatten over time/hidden dimensions, class projection | Same task-output contract test |
| Mask-aware imputation normalization and optional calendar marks | `models/TimesNet/Imputation.py` | Input channels are `[values | observed mask | calendar]`; mean and spread use observed values | `test_timesnet_imputation_matches_tslib_with_mask_aware_normalization` checks embedded normalized inputs before TimesBlock |

## Initialization and protocol

Parameters use native PyTorch initialization unless a configured initialization
policy is selected. The paper's block has two Inception convolutions with GELU
between them; the Transformer width and positional embedding are documented in
the source notes.

`configs/forecasting/ETTh1-TimesNet-Source.json` is an engineering/FedAvg example,
not a TimesNet paper-result configuration. The SMD anomaly configurations are
`configs/anomaly_detection/SMDAll-TimesNet-Historical.json` and
`SMDAll-TimesNet-Paper.json`: they keep the historical train/test-threshold
protocol and the separate validation-only protocol, respectively. The paper
reports SMD Adjusted F1 0.8581. The historical score is protocol-sensitive; report
the validation-only score separately and do not use smoke runs as paper results.
The ceiling-corrected full-budget runs score 0.857903 for the historical protocol
and 0.823999 for the validation-only protocol (seed 2021, 10 epochs); these are
recorded separately because their threshold populations differ.

TSLib also selects positive FFT bins and uses floor division for the period;
KairoTS documents the zero-DC and floor/ceiling differences in
[`explanation/notes.md`](../../explanation/notes.md). Therefore fixed-weight
numerical comparison is not a valid oracle when a selected `T/f` is nonintegral.
The dedicated period fixture checks the paper reshape, while the task contract
test checks output shapes and finite values against TSLib.
