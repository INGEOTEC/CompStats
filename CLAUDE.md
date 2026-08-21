# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CompStats implements a bootstrap-based evaluation methodology for statistically comparing the
performance of multiple algorithms/systems in a competition-style setting (e.g., comparing several
classifiers' predictions against the same held-out gold labels). It follows the `sklearn.metrics`
API convention: score functions take `(y_true, y_pred)` but return a rich object (`Perf`) instead of
a bare float, giving access to bootstrap standard error, confidence intervals, and pairwise
significance testing between systems.

## Language policy

All project artifacts must be written in English: function/variable names, comments, parameters,
docstrings, and any other generated text. This is a research project, so code clarity matters —
follow the existing docstring/comment style shown throughout `CompStats/` (see `metrics.py`'s
`@metrics_docs` pattern, and the `:param:`/`:type:`/doctest-style docstrings in `interface.py`,
`bootstrap.py`, `measurements.py`) rather than introducing a different documentation style.

Instructions from the user, in this conversation, may be given in either English or Spanish —
that does not change the above: respond and implement with English identifiers/comments/docs
regardless of the language the request was made in.

## Workflow

- Work is driven by GitHub issues: an issue is created describing the implementation to make, and
  Claude is asked to read that issue and implement it.
- Never implement directly on the main branch. If not already on a branch other than main, create
  a new branch first.
- The user typically works on a branch called `develop`. If already on `develop` (or another
  non-main branch), it's fine to implement there directly — no need to create a new branch just
  for that.

## Commands

Install the package and its dependencies (editable install, used by the devcontainer via
`.devcontainer/python.sh`):

```bash
pip install -e .
pip install -r requirements.txt
```

Run the full test suite (this is what VS Code's test explorer and `.vscode/settings.json` are
configured to use — despite CI still invoking `nosetests`, day-to-day development here uses pytest):

```bash
pytest CompStats
```

Run a single test file or test:

```bash
pytest CompStats/tests/test_interface.py
pytest CompStats/tests/test_interface.py::test_Perf_name
```

Run with coverage (mirrors what CI collects, excluding the `tests` package per `.coveragerc`):

```bash
coverage run -m pytest CompStats
coverage report
```

Build the Sphinx docs:

```bash
cd docs && make html
```

Note: `.github/workflows/test.yaml` (CI) builds the environment with conda and runs `nosetests`,
not pytest — this is legacy and inconsistent with local/devcontainer tooling. Don't be surprised if
CI config and local dev commands diverge; prefer `pytest` locally.

## Architecture

The package is small and organized around one core data flow: raw predictions → bootstrap resampled
statistic → derived comparisons/plots.

- **`bootstrap.py` — `StatisticSamples`**: the foundational primitive. Given a `statistic` callable
  (e.g. `accuracy_score`), it draws `num_samples` bootstrap resamples (with replacement) of the
  population and evaluates the statistic on each resample, optionally in parallel (`joblib`).
  Results for a named system are cached in `self.calls[name]` (a dict of name → ndarray of bootstrap
  samples). Bootstrap sample *indices* are cached per population size in `self._samples`, so multiple
  algorithms evaluated against the same `y_true` reuse the same resampling (this is what makes
  pairwise comparisons valid/paired). Supports `__sklearn_clone__` so `sklearn.base.clone` produces a
  fresh instance carrying over params (used heavily to create `Difference` objects from a `Perf`
  without recomputing bootstrap samples).

- **`interface.py` — `Perf` and `Difference`**: the main user-facing entry point (re-exported at
  package root). `Perf(y_true, *y_pred, name=..., score_func=..., error_func=..., **kwargs)` wraps
  one or more systems' predictions against shared ground truth. Exactly one of `score_func` /
  `error_func` must be set (asserted via XOR) — `score_func` implies bigger-is-better (`BiB=True`),
  `error_func` implies smaller-is-better (`BiB=False`). Internally holds a `StatisticSamples` keyed
  by system name; new predictions can be added later via `perf(y_pred, name=...)` (`__call__`).
  `Perf.difference(wrt=...)` produces a `Difference` instance (comparing every system against the
  best, or an explicit reference) whose `p_value()` is computed directly from the bootstrap
  distribution of paired differences — no parametric test assumptions. `Perf.plot()` /
  `Difference.plot()` render via seaborn `catplot`, with confidence intervals computed by
  `measurements.CI` passed as the `errorbar` callback.

- **`metrics.py`**: thin wrappers around `sklearn.metrics` functions (`accuracy_score`,
  `balanced_accuracy_score`, `top_k_accuracy_score`, `f1_score`, etc.). Each wrapper closes over the
  sklearn metric (plus its metric-specific kwargs like `average`, `normalize`) and constructs a
  `Perf` with that as `score_func`/`error_func`. The `@metrics_docs` decorator (from `utils.py`)
  injects the shared `Perf`-style docstring (params like `num_samples`, `n_jobs`, `use_tqdm`) into
  each wrapper automatically — when adding a new metric wrapper, follow this same
  `@metrics_docs(hy_name=..., attr_name=...)` + inner-function-closure pattern rather than duplicating
  docstrings.

- **`measurements.py`**: stateless helpers — `CI` (percentile bootstrap confidence interval), `SE`
  (bootstrap standard error), `difference_p_value`. Each accepts either a raw ndarray of bootstrap
  samples or a `StatisticSamples` instance (in which case it maps itself over `.calls`).

- **`performance.py`**: an alternative, more functional (non-`Perf`) API operating directly on a
  `pandas.DataFrame` (one gold column + one column per system) — `performance()`,
  `difference()`/`all_differences()`, and the `plot_performance*`/`plot_difference*` family, plus
  `*_multiple` variants for comparing several metrics at once (used for multi-metric competition
  reports: coefficient of variation, PPI, distance-to-best per metric). This module is older/more
  ad-hoc than `interface.py`'s `Perf`; new comparison features generally belong on `Perf`/`Difference`
  unless they specifically need the DataFrame-of-multiple-metrics shape.

- **`utils.py`**: `progress_bar` (tqdm wrapper, no-op if tqdm isn't installed or `use_tqdm=False`),
  `metrics_docs` (docstring-injecting decorator described above), and `dataframe()` (melts a `Perf`'s
  or `Difference`'s bootstrap samples into a long-format DataFrame for seaborn plotting).

### Key invariants to preserve when modifying this code

- Bootstrap resampling must stay *paired* across systems being compared — `StatisticSamples.samples`
  caches resample indices by population size `N` precisely so every system's bootstrap replicate `i`
  uses the same resampled indices. Don't introduce per-system independent resampling.
- `BiB` (Bigger is Better) must be threaded consistently: `score_func` → `BiB=True`, `error_func` →
  `BiB=False`. Sorting, `best`, and p-value sign logic throughout `interface.py`/`performance.py`
  depend on this flag rather than re-deriving it from the function.
- `sklearn.base.clone` / `__sklearn_clone__` is used to duplicate `Perf`/`StatisticSamples` instances
  while reusing already-computed bootstrap samples (e.g. `Perf.difference()`, `performance.difference`).
  Don't replace these with plain re-instantiation, as that silently redraws new bootstrap samples and
  breaks paired comparisons.
