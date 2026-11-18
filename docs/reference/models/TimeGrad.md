# TimeGrad

Rasul et al., “Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting,” ICML 2021, arXiv:2101.12072v2.
The implementation is in `src/kairots/models/TimeGrad/__init__.py`; its residual network is in `src/kairots/models/TimeGrad/Residual.py`.

## Equation-to-code map

| Operation | Implementation | Axes / evidence |
|---|---|---|
| Lagged target features and autoregressive LSTM state | `TimeGrad.recurrent_features`, `TimeGrad.forward` | channel-major lags and causal time axis; `test_timegrad_lags_are_strictly_causal_and_channel_major`, `test_timegrad_prediction_ignores_future_observations_and_scales_context_mean` |
| Forward diffusion and summed noise-prediction objective | `TimeGrad.training_loss` | prediction steps × target channels; `test_diffusion_objective_sums_vector_coordinates`, schedule checked in `test_timegrad_prediction_ignores_future_observations_and_scales_context_mean` |
| Reverse diffusion and joint sample trajectories | `TimeGrad.forward`, `TimeGrad.predict_noise` | output `[batch, horizon, target_channel, sample]`; `test_timegrad_prediction_ignores_future_observations_and_scales_context_mean` checks the output contract and future-observation isolation; the pinned author source is recorded below |

The paper configuration uses KairoTS's `Random` initializer, which preserves the native PyTorch initialization of each module; no separate initialization scheme is specified in the config.

## Model and paper configuration

TimeGrad predicts a joint distribution one step at a time.
An LSTM receives lagged target values, entity embeddings, and known calendar features.
Its state conditions a diffusion network that predicts noise for the current multivariate target.
Generated samples are fed back to the LSTM to produce the next step.
Training samples a diffusion step and Gaussian noise, then minimizes the paper's prediction-window squared noise error.
Inference reverses the diffusion process for each time step and retains joint trajectories for CRPS and CRPS-sum scoring.

The paper's main experiment uses context length equal to the prediction horizon, batch size 64, Adam with learning rate 0.001, 100 diffusion steps with a linear beta schedule from 0.0001 to 0.1, a two-layer LSTM with 40 hidden units, and 100 sample trajectories.
It uses a validation period equal in size to the test period to select the training duration.
Table 2 reports means and standard errors over ten runs.

### CRPS-sum formula and Table 2 scale conflict

Section 4.1 defines CRPS-sum by first summing each sampled forecast and target across all series, computing CRPS for that aggregate, and averaging over the forecast horizon.
`metrics/CRPSSum.py` follows that definition.
Table 2 reports Electricity as `0.0206 ± 0.001`, but the paper gives no normalization factor for this metric.
On the released Electricity archive, the paper-config runs produce raw CRPS-sum values in the thousands.
The benchmark therefore keeps the explicit paper formula, reports its scale discrepancy with Table 2, and does not claim a numerical reproduction or add an undocumented normalization to force a match.

The pinned author Trainer fills omitted training settings with 100 epochs, 50 batches per epoch, Adam weight decay `1e-6`, and OneCycleLR with maximum rate `1e-2`.
The paper's explicit batch size 64 and base learning rate `1e-3` take precedence over the Trainer defaults.
The paper does not state the validation selection score; the KairoTS paper preset selects the lowest validation CRPS-sum, the Table 2 metric, and reports final-epoch results separately.
The author Trainer instead uses validation denoising loss and does not select a checkpoint; the difference is explicit in the benchmark record.

The active representative matrix's candidate cell is the Table 2 Electricity row: 370 hourly series, 5,833 training steps, 24-step context and forecast, and reported CRPS-sum 0.0206 ± 0.001.
The pinned author settings use lags `[1, 24, 168]` and hourly/weekday Fourier features (four calendar channels), giving 192 history rows including lag context.
`GPCopula.ElectricityNIPS` reads the released 370-series [archive](https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/442bd4ffffa4a0fcf9ae7aa25db9632fbe58a7ea/datasets/electricity_nips.tar.gz) at revision `442bd4ffffa4a0fcf9ae7aa25db9632fbe58a7ea` (SHA-256 `b80a63a4963f487dc3b2a3be8cb17b3c1e54f71fddae9b5a12967b38c990c4cf`).
It reconstructs the pinned GluonTS `MultivariateGrouper` rules: train series align to the earliest start and use each series' mean for left/right padding; the rolling test series use zero for left padding.
The source checks exact train/test overlap, channel IDs, seven complete rolling dates and test overlap before writing the prepared multivariate timeline.
These grouping defaults are pinned to the official [GluonTS 0.8.0 grouper source](https://raw.githubusercontent.com/awslabs/gluonts/v0.8.0/src/gluonts/dataset/multivariate_grouper.py) as an explicit omission setting; the TimeGrad setup's `gluonts>=0.8.0` does not establish the authors' exact installed version.

The real-archive preparation produced 5,833 train rows followed by 168 test rows, with 5,450 train windows and seven each for validation and test.
The seven validation windows end at the training cutoff; the seven test targets use the released rolling dates.
This verifies input construction and split counts.
Exchange is another paper dataset (8 business-daily series, 6,071 training steps, horizon 30); the checked-in example uses its frequency-specific inputs but not its paper dataset release or split.

`configs/sample_forecasting/ElectricityNIPS-TimeGrad-Paper.json` is the paper preset.
It uses the pinned GP-Copula archive, 192 history rows, 24-step targets, seven validation and seven rolling test windows, batch size 64, 50 batches per epoch sampled with replacement from eligible train windows, 100 epochs, the approved author schedule omissions, 100 trajectories, and validation CRPS-sum selection.
Run ten seeds before comparing its mean and standard error with Table
2. The checked-in
`configs/sample_forecasting/ExchangeRate-TimeGrad.json` remains an engineering example: it reads the separate TSLib ExchangeRate CSV, uses a 32-step input (30 context steps plus lags 1 and 2), and spaced windows; it is not the paper preset.

## Paper and author-code differences

The method ledger records the full equation-to-code audit and the pinned author reference at [`zalandoresearch/pytorch-ts` commit `7860c9693d55b5c086867477cc33c89485ed0167`](https://github.com/zalandoresearch/pytorch-ts/tree/7860c9693d55b5c086867477cc33c89485ed0167).
Among the audited differences, the author trainer uses fixed epochs and final weights while the paper specifies validation-based epoch selection.
The selected benchmark protocol follows the paper; final-epoch metrics may be retained as a separate diagnostic.
The PI confirmed this choice on 2026-11-12 because the paper states validation-based training-duration selection explicitly, while the pinned trainer's fixed-epoch behavior is a separate historical protocol.
The paper does not give a maximum epoch count or patience.
Under the PI's prior approval to use pinned author settings for omissions, the paper preset uses the Trainer's 100-epoch default as its search cap and does not add a patience rule.
The paper also omits some frequency-dependent lag and projection details, so the benchmark records the approved pinned author settings used for those omissions.
The pinned author frequency helpers specify business-daily Exchange covariates as weekday/day-of-year Fourier features with lags `[1, 2]`.
They specify Electricity as hour/weekday Fourier features with lags `[1, 24, 168]`.
`data.calendar.features` supports the author's dynamic Fourier period (`max(field) + 1`) instead of substituting TSLib's normalized scalar feature; a focused formula test covers leap-year dates.
The engineering config uses the Exchange features but still uses TSLib's different dataset release and window protocol.

The ten-seed ElectricityNIPS run completed with the declared seeds 2021–2030, since the paper does not report its random seeds.
Validation-selected test CRPS-sum was `6263.0406 ± 155.8759` (mean ± sample standard error); final-round test CRPS-sum was `6324.9470 ± 264.8204`.
These raw values do not match Table 2's `0.0206 ± 0.001`.
The implementation retains the explicit raw paper formula and does not claim a numerical reproduction because the paper gives no normalization that reconciles the reported table scale.
Per-seed scores:

| Seed | Selected round | Selected test CRPS-sum | Final test CRPS-sum |
|---:|---:|---:|---:|
| 2021 | 60 | 7327.1055 | 8521.6387 |
| 2022 | 77 | 6534.1011 | 6557.1758 |
| 2023 | 70 | 6423.8525 | 6186.9736 |
| 2024 | 75 | 5609.6069 | 5758.7383 |
| 2025 | 36 | 6261.6904 | 6603.7476 |
| 2026 | 84 | 5873.4219 | 5732.9121 |
| 2027 | 82 | 6530.9048 | 6237.1460 |
| 2028 | 81 | 5850.2090 | 6071.7617 |
| 2029 | 44 | 5900.9102 | 5888.7358 |
| 2030 | 31 | 6318.6040 | 5690.6406 |

## Execution acceleration — 2026-11-12

The run uses all four RTX 4090 GPUs in parallel.
Seeds 2021–2030 remain the predeclared schedule, with one run record per seed; the paper configuration is unchanged.
Partial progress is not a paper score; compare selected-validation results with Table 2 only after all ten runs complete.
