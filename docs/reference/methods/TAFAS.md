# TAFAS test-time adaptation

TAFAS adapts a frozen multivariate forecaster on targets as they become
observable in a stride-one test stream. It changes only identity-initialized
calibration layers around the source model.

## Sources

- Kim et al., [Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation](https://arxiv.org/abs/2501.04970), AAAI 2025, arXiv v1, Eqs. 1–7 and Appendix Algorithm 1. The paper and rendered PDF are in the local paper library under `arxiv/10_48550_arxiv_2501_04970`.
- Official implementation: [kimanki/TAFAS at `139bf980`](https://github.com/kimanki/TAFAS/tree/139bf980671da4daad728a0fc21d8df508b9203d), especially `tta/tafas.py`.
- Paper/appendix and author-code differences are recorded in [`explanation/notes.md`](../../explanation/notes.md#tafas-adapts-through-the-prequential-path).

## Paper-to-code map

| Paper operation | KairoTS implementation | Shape or convention | Check |
|---|---|---|---|
| Remove each input channel's temporal mean, choose the highest-power channel and its dominant non-DC frequency; set `p = ceil(L / f*)` (PAAS, Eqs. 1–2) | `evaluation/TAFAS.py`, `TAFAS.period` | Input `[input_length, channels]`; each adaptive group has up to `p + 1` successive stride-one windows | `test_period_is_the_dominant_cycle_length` |
| Transform input and output with gated temporal affine residuals (GCM, Eq. 3) | `Calibration.forward`, `TAFAS.stream` | Per-channel `[length, length]` matrix and bias; `x + tanh(g) * (W x + b)` | `test_calibration_is_the_identity_at_initialization` |
| Update GCM parameters using partial and eligible full-target MSEs (Eqs. 4–6) | `TAFAS.stream`, nested `fit` | Partial targets are the leading `p` outputs of the first window; the full term uses the latest earlier group whose horizon has arrived; one optimizer step minimizes their sum | `test_first_mini_batch_matches_a_manual_derivation`, `test_every_horizon_that_has_passed_is_learned_from` |
| Replace predictions for targets that remain unobserved (Eq. 7) | `TAFAS.stream` | Every window in the group is adjusted, including the last; already-observed portions remain fixed | `test_unobserved_targets_never_reach_an_earlier_prediction` |
| Reveal targets only after their time step arrives | `TAFAS.revealed`, `TAFAS.evaluate` | A guard rejects early reads; test evaluation first records frozen scores, then scores adapted stream predictions | `test_unobserved_targets_never_reach_an_earlier_prediction`, `test_adaptation_changes_scores_and_restores_the_source_model` |

The source forecaster remains in evaluation mode and its parameters are restored
after scoring. Each client starts from its own source parameters and fresh,
identity calibration modules. `steps=0` skips adaptation and matches frozen
scores. Forecast channels must equal input channels, and target windows must have
stride one; calendar marks and separate target channels are unsupported.

## Configuration and evidence

The public component is `evaluation.TAFAS`. Defaults are Adam, learning rate
`1e-3`, weight decay `0`, gate value `0.01`, and one update at each eligible
adaptation point. See
[`ETTh1-DLinear-TAFAS.json`](../../../configs/forecasting/ETTh1-DLinear-TAFAS.json)
for a runnable one-seed example. The config uses the 0.6/0.2/0.2 split, DLinear,
and 30 centralized updates.

The recorded seed-941 pilot selected source round 28 and made 132 TAFAS updates:
frozen MSE/MAE 0.457399/0.455171 and adapted MSE/MAE 0.453564/0.453147. This is
an engineering pilot, not a reproduction of the paper's 0.451→0.442 DLinear
result: it uses plain DLinear without the official per-window NST normalization,
and only one seed was run.

The main text governs KairoTS's period ceiling, combined partial/full loss, and
adjustment of every still-unobserved prediction. The appendix prints floor
division, omits the last prediction in its adjustment loop, and places
`zero_grad` after `backward`; the pinned author code also uses separate loss
updates. These conflicts and the selected main-text interpretation are explicit
in the linked notes. The final shortened group is skipped if its partial targets
have not yet arrived.
