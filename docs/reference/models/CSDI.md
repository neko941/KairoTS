# CSDI

Tashiro et al., “CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation,” NeurIPS 2021, arXiv:2107.03502v2.
The implementation follows the paper's diffusion equations and the pinned author implementation for dataset details omitted from the paper.

## Equation-to-code map

| Operation | Implementation | Axes / evidence |
|---|---|---|
| Forward process `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon` and epsilon-prediction MSE (CSDI Sec. 3; resolved with DDPM Eqs. 4–6) | `CSDI.training_loss` | `[batch, time, feature]`; `test_csdi_forward_noising_matches_resolved_ddpm_formula`, `test_diffusion_objective_sums_vector_coordinates` |
| Conditional score input from noisy targets, observed values/mask, and time features | `CSDI.predict_noise`, `conditional_mask` | observed values are clamped; time is the middle axis; `test_csdi_masks_native_missing_values_and_preserves_observed_zeros`, `test_csdi_timestamp_covariate_and_known_pattern` |
| Reverse DDPM posterior update and trajectory generation (DDPM Eq. 11) | `CSDI.forward` | `[batch, time, feature, sample]`; `test_csdi_reverse_chain_matches_ddpm_equations` |
| Bounded trajectory chunks preserve all configured samples | `CSDI.forward` | at most 128 batch-sample pairs per network call; `test_csdi_samples_in_chunks_without_changing_output_contract` |

The CSDI paper's printed diffusion expressions conflict with the cited DDPM equations; the implementation follows DDPM and records that decision below.
The default `Random` initializer preserves each PyTorch layer's constructor initialization; the paper configuration adds no separate initialization policy.

## PhysioNet 2012 imputation

`PhysioNet2012CSDI` reads the released Set-A cohort only.
It uses the ordered 35 time-varying attributes from the author parser, buckets measurements by integer hour, keeps the last value for each attribute in an hour, and constructs 48-hour windows.
Empty patients remain in the 4,000-record cohort.
Each cross-validation fold shuffles patient indices with seed 1, assigns the corresponding 20% to test, reshuffles the remainder with seed 1, and uses the first 70% of all records for training and the rest for validation.
This gives 2,800 / 400 / 800 records.

The historical config reproduces the author parser's normalization: each feature's mean and population standard deviation are computed from every observed Set-A value before the fold split.
The clean config fits the scaler on the training patients only.
Keep the two tracks separate in reports.
The task hides `floor(0.1 × observed positions)` per sample, as the parser does.
Inference uses 100 draws.
Inference evaluates trajectories in bounded chunks (at most 128 batch-sample pairs at once) to fit the paper's 100 draws on the available GPU; this changes random-number ordering, not the per-trajectory reverse-diffusion rule.
The pinned trainer uses the final fixed-budget weights, so use `final_metrics` for this reproduction; `best_val_metrics` is diagnostic only.
For the five-fold fixed-budget benchmark, train with validation/test metrics disabled, then evaluate the completed fit artifact once on the held-out patients.
Because panel data has no timestamps, the fit-to-evaluate initializer verifies that the source train-sample ranges and target test-sample ranges are disjoint.

```sh
python -m kairots prepare --config configs/tashiro2021-csdi/Table1-CSDI.json
python -m kairots --config configs/tashiro2021-csdi/Table1-CSDI.json
```

Select folds 1–4 with `--data.split.fold 1` through `--data.split.fold 4`.
Use the clean config for the separate train-only normalization benchmark.
The current `PhysioNet2012` mortality-classification source remains separate from this imputation adapter.

### Execution smoke (not a paper result)

One historical-normalization run completed one training update and evaluated all 100 requested trajectories.
Test NormalizedCRPS was `0.7168449` and MedianMAE was `0.6305870`; elapsed time was `5,552.3` seconds.
This checks the 100-trajectory execution path only.
It is not the trained five-fold paper benchmark and is not comparable to the paper's multi-epoch aggregate.

The clean and historical config variants also completed one-update, 100-trajectory smokes on a dedicated RTX 4090.
Clean-track test SampleCRPS / NormalizedCRPS / MedianMAE were `0.4327649 / 0.6309572 / 0.5713896` (2,526.95 s); historical-track values were `0.4345828 / 0.6356996 / 0.5731674` (2,519.48 s).
These remain execution checks, not trained five-fold paper results.

## Trained five-fold historical-normalization benchmark

The historical-normalization paper config completed five folds at 200 centralized iterations per fold.
Each final-budget checkpoint was evaluated once on its held-out patients using all 100 configured trajectories.
Values below are pooled held-out test metrics (one client per fold); aggregate is mean ± sample standard error over the five fold scores.

| Fold | NormalizedCRPS | MedianMAE | SampleCRPS |
|---:|---:|---:|---:|
| 0 | 0.239241 | 0.220317 | 0.162315 |
| 1 | 0.234527 | 0.218179 | 0.160360 |
| 2 | 0.244625 | 0.221249 | 0.164146 |
| 3 | 0.241531 | 0.221071 | 0.163791 |
| 4 | 0.239630 | 0.218346 | 0.162253 |
| Mean ± sample SE | 0.239911 ± 0.001649 | 0.219832 ± 0.000660 | 0.162573 ± 0.000672 |

Folds 0–3 ran on RTX 4090 GPUs and fold 4 on an RTX 3080.
Python 3.14.4, PyTorch 2.11.0, NumPy 2.4.4, Polars 1.40.0 and Ray 2.55.0 matched across hosts; the CUDA builds differed (cu126 and cu130).
This hardware and CUDA variation is recorded as a benchmark environment detail.
The separate train-only-normalization benchmark is reported independently below.

The paper's Table 1 reports NormalizedCRPS `0.238 ± 0.001` for PhysioNet at the 10% target/masking setting.
The historical benchmark mean `0.239911 ± 0.001649` is within one combined standard error of that value.
The paper reports five trials but does not identify them as patient folds; this benchmark uses five disjoint held-out patient folds.
The scores are numerically consistent, but this difference means the aggregate is not an exact reproduction of the paper's trial protocol.

## Paper and source differences

The paper delegates the score model to DDPM; the implementation uses the cited DDPM equations where CSDI's printed diffusion expressions conflict.
The official parser is pinned at [`ermongroup/CSDI` commit `7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`](https://github.com/ermongroup/CSDI/tree/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27).
Its PhysioNet parser uses the full-cohort normalizer before splitting; the clean track is a separate leakage-free evaluation choice.
For an even sample count, `MedianMAE` uses PyTorch's lower median, matching the author code's `median` operation.

### Separate train-only-normalization results

The clean track completed the same five-fold, 200-iteration fit and 100-trajectory evaluation protocol, with normalization fitted on training patients only.
Its fold scores and mean ± sample standard error are:

| Fold | NormalizedCRPS | MedianMAE | SampleCRPS |
|---:|---:|---:|---:|
| 0 | 0.240062 | 0.221834 | 0.163549 |
| 1 | 0.234215 | 0.217315 | 0.160069 |
| 2 | 0.240668 | 0.218472 | 0.161957 |
| 3 | 0.242420 | 0.223423 | 0.164774 |
| 4 | 0.240527 | 0.219280 | 0.162510 |
| Mean ± sample SE | 0.239578 ± 0.001399 | 0.220065 ± 0.001121 | 0.162572 ± 0.000789 |

All five clean folds ran on RTX 4090 GPUs with the cu126 PyTorch build.
The historical-normalization and train-only-normalization results are separate tracks; compare them as distinct preprocessing protocols.
The one-update execution smokes above are not trained paper results.
