# Pitfalls

Traps this code has hit before, each with what to do instead.
The ids stay fixed; a new pitfall takes the next free id.

## KT2

**When:** printing non-ASCII (tree glyphs, arrows) from the KairoTS Python with -I.

**Trap:** -I ignores PYTHONIOENCODING, so stdout stays cp1252 and print raises UnicodeEncodeError.

**Do:** call sys.stdout.reconfigure(encoding="utf-8") first.

## KT3

**When:** writing comments, docstrings, docs or design notes in KairoTS.

**Trap:** text that is not about the code goes stale and misleads readers: tool tags, who wrote or approved something, plan or milestone talk, dated log headings.

**Do:** keep only what the code is and does, and how it relates to its references (papers, TSLib); state a known limit as a plain sentence in docs/explanation/notes.md.

## KT4

**When:** checking the forecasting parity run.

**Trap:** test MSE is scaled by default (evaluation.inverse false, BottomUp excepted) and DLinear keeps PyTorch's default init (the official 1 / input_length start is Average with body-weight patterns), so older references (raw 9.374014854431152, scaled 0.23660719394683838) no longer apply.

**Do:** compare the scaled reference 0.2682070732116699 (default) and the raw one 10.65915584564209 with --evaluation.inverse true; the final round is the last generalization_test_pooled_mse in the output (metric keys are <population><generalization|personalization>_<cohort><phase>_...).

## KT6

**When:** changing split, scaler or store code and re-running smoke on an existing datasets/ root.

**Trap:** the prepared keys hash the files that define the source, partition, store, split and scaler classes and their bases (Parquet.code), but not helper modules they import (data/calendar.py, data/download.py, numpy helpers); a change there reuses a stale cache, and every hashed change leaves the old prepared directory behind.

**Do:** after changing an unhashed helper, delete datasets/<family>/prepared (or run scripts/clean.py --prepared); run scripts/clean.py --prepared now and then to reclaim disk.

## KT7

**When:** rewriting a reduction in a scaler, loss or metric (for example std as sqrt(nanmean of squares) to skip NaN).

**Trap:** the new formula is mathematically equal but differs in the last float bits, so the scaled parity value drifts (for example 0.236607164 instead of 0.23660719394683838) while the raw one may still match.

**Do:** keep the original torch operation on the path without NaN and branch only when NaN is present; check parity with --data.store.root on a fresh directory, since a cached split hides scaler changes (KT6).

## KT8

**When:** splitting or windowing month-start (or quarterly) dates.

**Trap:** calendar months differ in length, so a timedelta step taken as the smallest gap marks every 31-day month as a gap and leaves no complete windows.

**Do:** CompleteWindowRatio._grid counts month starts in months; reuse it for any new continuity check.

## KT9

**When:** monkeypatching a module-level function in a component module (kairots.data.stores.Parquet.code).

**Trap:** registry.get rebinds the class on its category package, so the dotted path kairots.data.stores.Parquet resolves to the class, not the module, and monkeypatch fails with "has no attribute".

**Do:** patch sys.modules["kairots.data.stores.Parquet"] (the module object) instead of the dotted string.

## KT10

**When:** editing docs/conf.py, or writing a Python string literal with a backslash escape through a shell heredoc.

**Trap:** scripts/format.py never compiles docs/conf.py, and the heredoc path can land a real newline inside the literal; the gate then fails with docs=2 (Sphinx ConfigError).

**Do:** edit docs/conf.py with the Edit tool and run `.conda/python.exe -m py_compile docs/conf.py` before the gate.

## KT11

**When:** subclassing a registered component (policy, strategy, model) whose base declares MODELS, TASKS or STRATEGIES.

**Trap:** registry info reads these class attributes from the component's own file without following inheritance, so a subclass that relies on the inherited value loses the check (a subclass can otherwise accept unsupported models).

**Do:** restate MODELS, TASKS or STRATEGIES in every registered subclass.

## KT12

**When:** filtering round or summary metrics by phase in a test or script.

**Trap:** keys read `<population><scoring>_<cohort><phase>_<pooled|mean_client>_<metric>` (generalization_test_pooled_mse, unseen_personalization_cohort0_test_...), so startswith("test_") matches nothing and a leakage comparison silently includes test scores.

**Do:** match the phase segment (`"test_" in key`) and the `unseen_` prefix, or build the full key.

## KT13

**When:** resetting a client's optimizer state between rounds (FML meme model, any per-round restart).

**Trap:** an empty optimizer_state also drops param_groups, so the next round trains at the initial learning rate and a run-span schedule is silently undone (the scheduler's state does not rewrite param_groups).

**Do:** keep param_groups and clear only "state"; for a schedule that restarts each round use Cosine `span: "round"`.

## KT14

**When:** building a fresh scheduler on an optimizer restored from a client's saved state (Cosine `span: "round"`).

**Trap:** optimizer.load_state_dict restores the last update's annealed lr, and CosineAnnealingLR's first step keeps the current lr, so every update after the first trains at `minimum_rate` (0 by default); SCAFFOLD option II then divides by that rate.

**Do:** reset each param group's lr to its `initial_lr` before building a per-update scheduler; divide by the summed step sizes, never by the final lr.

## KT15

**When:** adding a registered component (a model, metric, task) under src/kairots.

**Trap:** tests/unit/test_registry.py and tests/unit/test_metrics.py pin the exact sorted name tuples per category and per task, so a new component fails them on the server gate.

**Do:** add the new name to those tuples in the same change; also a `TASKS` constant decides which per-task tuple it joins.

## KT16

**When:** subclassing a component (a metric, a model) that defines its own `__init__(self, *, config=None)`.

**Trap:** decorating the subclass with `@dataclass` generates a new `__init__` from the fields, which shadows the parent's and drops `config`, so the registry's `Metric(config=...)` raises TypeError at run time while `Metric()` in unit tests still works.

**Do:** leave the subclass undecorated (only its nested `Config` needs `@dataclass`, subclassing the parent `Config`); a config listing the component makes `scripts/smoke.py` catch it.

## KT17

**When:** recording dataset family for pretrained or zero-shot runs.

**Trap:** marking every Composite unknown also hides ETDatasetHour's homogeneous ETTh1/ETTh2 family and wrongly enforces cross-dataset calendar checks.

**Do:** inspect explicit MEMBERS; use their common family only when all members agree; keep heterogeneous composites and groups unknown, and retain same-client temporal checks across families.

## KT18

**When:** adding a distribution or sample head to inverse scoring.

**Trap:** a new transformation branch is unreachable unless its HEAD also joins Frozen.evaluate's restore predicate; Gaussian standard deviation must be scaled without adding the mean.

**Do:** update the restore predicate and transform together; test raw GaussianNLL minus scaled GaussianNLL against the log-scale Jacobian, and preserve the last sample axis for joint CRPS.

## KT19

**When:** fitting CSDI or TimeGrad.

**Trap:** Gradient calls forward and would train through sampling; their forward methods are inference-only and must never receive held-out values.

**Do:** select Denoising in fp32; training_loss owns self-supervised masks or teacher-forced past observations, and Frozen calls forward with prepared inputs only.

## KT20

**When:** adding a choice that a config selects by name (an optimizer, a loss, a scheduler, a regime).

**Trap:** a name table (`optimizers/__init__.py` OPTIMIZERS) or a torch class hard-coded in a strategy or protocol (Adapted's Adam, Ditto's SGD) cannot carry settings, so FML's momentum and weight decay were inexpressible; it also skips the registry and the generated docs page.

**Do:** one file per component with a nested Config and a build method, as in schedulers/; the category's `__init__.py` is only the registry surface (docs/reference/conventions.md).
A table is acceptable only for option-free functions whose options already pass through (INITIALIZERS).

## KT21

**When:** a component only works with certain other settings (a learner, an optimizer, a protocol, no groups).

**Trap:** a branch on the component's name in config/schema.py __post_init__ keeps the rule away from the component, so rules drift and every new method needs a central edit.

**Do:** declare the rule on the component as a literal constant that the schema reads generically (`REQUIRES`, path to allowed values; it checks the config and never overwrites a value).
Use a `validate` hook only for a relation between values.
Never add `if self.x.name == "..."` to config/schema.py.

## KT22

**When:** a loss and a metric compute the same quantity (Return, SampleCRPS, GaussianNLL).

**Trap:** two copies of the computation drift, and evaluation runs both (Frozen ran the Battery rollout twice per batch).

**Do:** one class with `ROLES`, or at least one formula function shared by both and computed once per batch.

## KT23

**When:** fitting preprocessing (a scaler) for a regime that releases training data over rounds (Continual).

**Trap:** fitting on the whole first task uses examples released in later rounds; the future-task leakage test still passes, while poisoned round-1 examples moved client scaler means from 0.12 to 499.57.

**Do:** fit only on what the first round releases (`released(current_iteration=0, ...)`, in the worker's sorted order); a leakage test poisons future-round data, not only future-task data.

## KT24

**When:** a model chooses between paths (forecast vs reconstruction head, extended vs kept time axis).

**Trap:** inferring the task from shapes (TimesNet treated input_length == output_length as reconstruction) silently drops a layer from a valid 96 to 96 forecast.

**Do:** select by the head or task name, never by comparing lengths; add an equal-length test against the reference.

## KT25

**When:** the orchestrator or a store reads a setting that a client group can override (split holdouts, scaler, source).

**Trap:** reading only the root `experiment.data` ignores `groups[g].data`, so a group's held-out clients trained and `unseen_clients` was empty.

**Do:** resolve per group (`experiment.member(group=g)`) and map partition ids to run ids; every such feature gets a test with groups.

## KT26

**When:** applying several config changes (CLI overrides, group merges).

**Trap:** validating after each partial step rejects a valid final config (switching to FedBABU failed before finetune_epochs was applied).

**Do:** apply every change, then validate once; keep type parsing and unknown-key errors per change.

## KT27

**When:** citing where a method, dataset, protocol or default comes from, in code, docs or notes.

**Trap:** citing a port (another library that reimplemented it) hides what the original does, and a deviation from the port is not a deviation from the method.

**Do:** cite the paper (authors, venue, year) and, when one exists, the original repository as `owner/repo` at a pinned commit; record a deviation against that original in docs/explanation/notes.md.

## KT28

**When:** a config selects components inside another component, such as an optimizer, replay buffer or online regime.

**Trap:** checking only a list of top-level namespaces silently ignores nested `TASKS`, `REQUIRES` and `validate` contracts.

**Do:** walk nested component selections through the same validation path; test nested restrictions and keep existing config resolutions unchanged.

## KT29

**When:** a reference comparison depends on optional checkout files, dependencies or raw data.

**Trap:** a skipped comparison can make a green test run appear to establish reference parity.

**Do:** set `KAIROTS_REQUIRE_REFERENCES=1` in the verification gate, fail required reference skips and retain the complete skip summary.
