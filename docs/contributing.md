# Contributing

## What gets added

- Methods (models, strategies, metrics, ...) implement a published, peer-reviewed paper
or a standard baseline.
Cite the paper in the class docstring.
- Add a component by adding its file; see [Customization](customization/index.md).
Copy the reference file named on its page.
- Add a runnable config under `configs/<task>/` only for a combination that has been
run end to end.

## Faithful methods

Each method reproduces its original paper exactly, with nothing added and nothing left out.
Where the paper is silent, the official code decides.
Unavoidable deviations are listed in [design notes](explanation/notes.md).
A new idea is its own component and never changes a reproduced method.

A config that reproduces a published setting declares it, and the docs list it under [Paper settings covered](reference/reproductions.md):

```json
"reproduces": {
  "paper": "arXiv:2205.13504v3",
  "table": "Table 2",
  "cells": {"ETTh1 96 MSE": 0.375, "ETTh1 192 MSE": 0.405},
  "code": "https://github.com/cure-lab/LTSF-Linear/tree/<commit>"
}
```

Every value elsewhere in that config comes from the paper, or from the official code where the paper is silent.
A sweep (horizons, datasets) lists one cell per run, and each run overrides only the swept field.
A config without the block is an example of a framework feature and makes no paper claim.

## Rules

- Follow the [conventions](reference/conventions.md): file and class names, the
glossary, keyword-only parameters, complete type hints.
`scripts/check_style.py` checks them.
- Raw data under `datasets/<Family>/raw/` is read-only.
Run outputs go to `runs/`; neither is tracked.
- Record each design decision without dated headings or process chatter in
[design notes](explanation/notes.md) when you make it, newest first.
- Update the docs page of a component kind when its contract changes.
- Read [pitfalls](reference/pitfalls.md) before adding a component; add one when a
trap is found.

## Before you finish

Run the gate from the repository root; CI (`.github/workflows/tests.yml`) runs the same steps.

```bash
python scripts/format.py        # ruff fix + format, then the style checks (--check: report only)
python -m pyright               # src, tests, scripts
python -m pytest -q
python scripts/docs.py          # Sphinx; warnings are errors
python scripts/smoke.py         # every config for one iteration (downloads data)
python scripts/leakage.py       # training is identical with a poisoned test set
```

Smoke includes pretrained examples: it runs their source configurations first, reuses each source once, and passes the resulting temporary checkpoint to the target.
Checkpoints and runs are removed when smoke finishes.
A missing source, failed source run or missing checkpoint fails the dependent example.
Models that draw samples (`model.samples`) draw one sample in smoke, so smoke checks the sampling path, not the configured sample count.

`python scripts/clean.py` deletes regenerable caches; it never touches `runs/`, `datasets/` or `.conda/`.
`--prepared` also deletes the prepared datasets: their keys include the data-path code, so each code change leaves the old ones unused.
