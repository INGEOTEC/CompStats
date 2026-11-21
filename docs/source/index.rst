.. _CompStats:

====================================
CompStats
====================================

.. image:: https://img.shields.io/badge/GitHub-CompStats-black?logo=github
        :target: https://github.com/INGEOTEC/CompStats

.. image:: https://github.com/INGEOTEC/CompStats/actions/workflows/test.yml/badge.svg
		:target: https://github.com/INGEOTEC/CompStats/actions/workflows/test.yml

.. image:: https://coveralls.io/repos/github/INGEOTEC/CompStats/badge.svg?branch=develop
		:target: https://coveralls.io/github/INGEOTEC/CompStats?branch=develop

.. image:: https://badge.fury.io/py/CompStats.svg
		:target: https://badge.fury.io/py/CompStats

.. image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/compstats-feedstock?branchName=main
	    :target: https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=20297&branchName=main

.. image:: https://img.shields.io/conda/vn/conda-forge/compstats.svg
		:target: https://anaconda.org/conda-forge/compstats

.. image:: https://img.shields.io/conda/pn/conda-forge/compstats.svg
		:target: https://anaconda.org/conda-forge/compstats

Version |compstats_version|.

.. note::

   This is the **developer reference** of **CompStats**: the architecture of the package and the
   complete API. Installation, tutorials and the description of the methodology for users are
   published on the project site, https://ingeotec.github.io/CompStats/.

Introduction
====================================

Collaborative competitions (challenges) evaluate systems on a single held-out test set whose gold
standard is kept by the organizers. The systems are ranked by a performance measure computed on that
set. The organizers have access only to the predictions, to a fixed dataset size and to a small
number of submissions; consequently, the classical statistical tests that require access to the
algorithm, or that require performance estimates on several datasets, do not apply.

**CompStats** implements a bootstrap-based methodology that complements the ranking with standard
errors, confidence intervals, significance tests of paired differences and corrections for multiple
comparisons. The same setting covers any comparison of supervised-learning systems on a shared
test set, whether the task is classification or regression. The methodology is described in
*Analysis of systems' performance in natural language processing competitions*
(`doi:10.1016/j.patrec.2024.03.010 <https://doi.org/10.1016/j.patrec.2024.03.010>`_), which extends
*Comparison of Classifiers in Challenge Scheme*
(`doi:10.1007/978-3-031-33783-3_9 <https://doi.org/10.1007/978-3-031-33783-3_9>`_).

Statistical background
====================================

Each concept below is stated with its formula and the symbol of the package that computes it.

**Setting.** Let :math:`y` be a gold standard of size :math:`n`, let
:math:`\hat{y}_1, \ldots, \hat{y}_m` be the predictions of :math:`m` systems, and let
:math:`s(y, \hat{y})` be a performance measure. A measure is either bigger-is-better (a score) or
smaller-is-better (an error); the flag ``BiB`` carries that direction.

**Bootstrap.** :math:`B` samples (``num_samples``, default 500) of :math:`n` indices are drawn with
replacement from :math:`\{1, \ldots, n\}`. The measure is evaluated on every sample for every
system, which gives the bootstrap distribution :math:`s_j^{*1}, \ldots, s_j^{*B}` of system
:math:`j` (:py:class:`~CompStats.bootstrap.StatisticSamples`). The **same** index samples are used
for all the systems, which is what makes the comparisons paired.

**Standard error.** The standard deviation of the bootstrap distribution,

.. math::

   \widehat{\mathrm{SE}}_j = \sqrt{\frac{1}{B} \sum_{b=1}^{B} \left(s_j^{*b} - \bar{s}_j^{*}\right)^2},
   \qquad \bar{s}_j^{*} = \frac{1}{B} \sum_{b=1}^{B} s_j^{*b},

computed by :py:func:`~CompStats.measurements.SE` and exposed as the ``se`` property of
:py:class:`~CompStats.interface.Perf`.

**Confidence interval.** The percentile interval

.. math::

   \left[ q_{\alpha/2}(s_j^{*}),\; q_{1-\alpha/2}(s_j^{*}) \right],

where :math:`q_p` is the :math:`p`-th quantile of the bootstrap distribution and
:math:`\alpha = 0.05` by default. It is computed by :py:func:`~CompStats.measurements.CI` and
exposed as the ``ci`` property of :py:class:`~CompStats.interface.Perf`.

**Paired difference.** For a reference system :math:`A` (by default the best one) and another
system :math:`B`, the bootstrap distribution of

.. math::

   \delta^{*} = s^{*}(y, \hat{y}_A) - s^{*}(y, \hat{y}_B)

is computed on the same samples. The sign is flipped for smaller-is-better measures, so that a
positive difference always means that the reference is better. It is built by
:py:meth:`~CompStats.interface.Perf.difference`, which returns a
:py:class:`~CompStats.interface.Difference`.

**Significance.** Let :math:`\delta(x)` be the difference observed on the test set. The p-value of
the null hypothesis :math:`H_0`: "the reference is not better" is estimated as the fraction of
bootstrap differences that are at least :math:`2\delta(x)`,

.. math::

   \hat{p} = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}\left[\delta^{*b} \geq 2\delta(x)\right],

(:py:meth:`~CompStats.interface.Difference.p_value`), following Berg-Kirkpatrick, Burkett and
Klein (2012). The bootstrap distribution is centred on :math:`\delta(x)` rather than on zero,
hence the factor 2.

**Multiple comparisons.** Comparing the best system against the other :math:`m - 1` systems tests
several hypotheses at once. The p-values can be adjusted with any method of
:py:func:`statsmodels.stats.multitest.multipletests` through the ``correction`` argument (for
example ``'bonferroni'``, ``'holm'`` or ``'fdr_bh'``). When several measures are evaluated, each
measure is corrected as its own family.

**Competitiveness of a competition.** The legacy functional API
(:py:func:`~CompStats.performance.performance_multiple_metrics`) summarizes the competitiveness of
a competition with the coefficient of variation of the systems' performance,

.. math::

   CV = 100 \, \frac{s_x}{\bar{x}},

where :math:`s_x` is the sample standard deviation and :math:`\bar{x}` the mean of the systems'
performance, and with the distance between the winner and the system in the middle of the table.
The potential for improvement of a task is assessed separately with the possible percentage
improvement, :math:`PPI = 100\,(1 - x_{best})`, for bigger-is-better scores bounded by 1.

Architecture
====================================

**CompStats** is organized around one core data flow: raw predictions from one or more
systems are turned into bootstrap-resampled statistics, from which derived comparisons and plots
are computed. This section gives a map of how the modules and classes relate to each other before
diving into the per-function API references that follow.

Data flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a set of systems' predictions and a shared ground truth (:math:`y`), the same bootstrap
resample indices are drawn once per population size and reused across every system being compared
-- this pairing is what makes the pairwise comparisons statistically valid. A performance statistic
(e.g. accuracy, F1) is evaluated on each resample for each system, producing an empirical
distribution per system. Everything else -- standard errors, confidence intervals, and pairwise
significance testing between systems -- is derived from these bootstrap distributions rather than
from parametric assumptions.

Module and class relationships
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

    classDiagram
        class StatisticSamples {
            statistic: Callable
            num_samples: int
            n_jobs: int
            BiB: bool or ndarray[bool]
            calls: dict~str, ndarray~
            __call__(*args, name)
            __sklearn_clone__()
        }
        class Perf {
            y_true: ndarray
            predictions: dict~str, ndarray~
            func: Callable or list~Callable~
            BiB: bool or list~bool~
            statistic_samples: StatisticSamples
            best
            difference(wrt)
            dataframe()
            plot()
            __call__(y_pred, name)
        }
        class Difference {
            statistic_samples: StatisticSamples
            statistic: dict
            best: str or ndarray
            p_value(right, correction)
            dataframe()
            plot()
        }
        class MetricsWrappers {
            <<CompStats.metrics>>
            accuracy_score(y_true, *y_pred)
            balanced_accuracy_score(y_true, *y_pred)
            f1_score(y_true, *y_pred)
            ...
            .measure(**kwargs) Callable
        }
        class PerformanceModule {
            <<CompStats.performance>>
            performance(dataframe, score)
            difference(statistic_samples)
            all_differences(statistic_samples)
            plot_performance(statistic_samples)
            plot_difference(statistic_samples)
        }
        class Measurements {
            <<CompStats.measurements>>
            CI(samples, alpha)
            SE(samples)
            difference_p_value(samples, BiB)
        }

        Perf "1" *-- "1" StatisticSamples : owns
        Difference "1" *-- "1" StatisticSamples : owns (cloned)
        Perf ..> Difference : difference() clones\nvia sklearn.base.clone
        MetricsWrappers ..> Perf : constructs, tagging\nfunc.BiB
        PerformanceModule ..> StatisticSamples : operates on directly
        Measurements ..> StatisticSamples : reads .calls

- **StatisticSamples** (:py:mod:`CompStats.bootstrap`) is the foundational primitive: given a
  ``statistic`` callable, it draws ``num_samples`` bootstrap resamples of the population and
  evaluates the statistic on each one, optionally in parallel via ``n_jobs``. Results for a named
  system are cached in ``calls`` (name -> ndarray of bootstrap samples); resample *indices* are
  cached per population size so every system sharing a population reuses the same resampling.
  ``__sklearn_clone__`` lets :py:func:`sklearn.base.clone` produce a fresh instance that carries over
  parameters -- used to build a :py:class:`~CompStats.interface.Difference` from a
  :py:class:`~CompStats.interface.Perf` without recomputing bootstrap samples.

- **Perf** (:py:mod:`CompStats.interface`) is the main user-facing entry point. It wraps one or more
  systems' predictions against shared ground truth, holding one :py:class:`~CompStats.bootstrap.StatisticSamples`
  instance keyed by system name. ``func`` is a single callable or a list of callables (a
  multi-measure ``Perf``); direction (bigger-is-better vs. smaller-is-better) comes from each
  callable's own ``BiB`` attribute when present (tagged by a :py:mod:`CompStats.metrics` wrapper's
  ``.measure`` factory), falling back to the constructor's ``BiB`` default otherwise --
  ``BiB`` is the single source of truth for direction; there is no separate score/error split.
  ``Perf.difference(wrt=...)`` produces a ``Difference`` comparing every system against the best (or
  an explicit reference).

- **Difference** (:py:mod:`CompStats.interface`) is created from ``Perf.difference()`` via
  :py:func:`sklearn.base.clone`, reusing the already-computed bootstrap samples rather than
  recomputing them. Its ``p_value()`` is computed directly from the bootstrap distribution of paired
  differences, with no parametric test assumptions.

- **CompStats.metrics** wrappers (``accuracy_score``, ``balanced_accuracy_score``,
  ``top_k_accuracy_score``, ``f1_score``, ...) are factories that close over an
  :py:mod:`sklearn.metrics` function (plus its metric-specific keyword arguments) and construct a
  ``Perf`` with that closure as ``func``, tagging it with ``.BiB`` so direction travels with the
  callable. The shared ``@metrics_docs`` decorator injects the common ``Perf``-style docstring into
  each wrapper.

- **CompStats.performance** is an alternative, more functional API operating directly on a
  ``pandas.DataFrame`` (one gold column plus one column per system): ``performance()``,
  ``difference()``/``all_differences()``, and the ``plot_performance*``/``plot_difference*`` family,
  plus ``*_multiple`` variants for comparing several metrics at once. It predates ``Perf`` and is
  intentionally not composed with ``Perf``/``Difference``.

- **CompStats.measurements** provides stateless helpers -- ``CI`` (percentile bootstrap confidence
  interval), ``SE`` (bootstrap standard error), and ``difference_p_value`` -- each accepting either a
  raw ndarray of bootstrap samples or a ``StatisticSamples`` instance, in which case it maps itself
  over ``.calls``.

Invariants to preserve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   **Bootstrap resampling must stay paired across systems.**
   ``StatisticSamples`` caches resample indices by population size precisely so every system's
   bootstrap replicate *i* uses the same resampled indices. Do not introduce per-system independent
   resampling -- it would invalidate every pairwise comparison.

.. note::
   **BiB (Bigger is Better) must be threaded consistently, never re-derived.**
   A callable's own ``BiB`` attribute wins when present; otherwise ``Perf``'s ``BiB`` constructor
   argument is the default. Sorting, ``best``, and p-value sign logic throughout
   :py:mod:`CompStats.interface`/:py:mod:`CompStats.performance` read this flag rather than
   re-deriving direction from which argument a function was passed as.

.. note::
   **Bootstrap samples must be reused via** ``__sklearn_clone__``, **not re-instantiation.**
   :py:func:`sklearn.base.clone` duplicates ``Perf``/``StatisticSamples`` instances while carrying
   over already-computed bootstrap samples (e.g. ``Perf.difference()``,
   ``CompStats.performance.difference``). Plain re-instantiation silently redraws new bootstrap
   samples and breaks paired comparisons.

Development
====================================

Install the package with its test dependencies and the documentation dependencies:

.. code:: bash

   pip install -e ".[test]"
   pip install -r docs/requirements.txt

Run the test suite:

.. code:: bash

   pytest CompStats

Build the developer reference; the HTML is written to ``docs/build/html/index.html``:

.. code:: bash

   python -m sphinx -n -W --keep-going -b html docs/source docs/build/html

The same strict build runs in continuous integration, and any warning (for example, a
cross-reference that does not resolve) makes it fail.

Releases are published by running the workflow ``.github/workflows/publish-pypi.yml`` manually
on ``main``. Creating a tag or a GitHub release does not publish the package. The version gate
``scripts/check_package_versions.py`` verifies that the version is a single step ahead of the one
published on PyPI.

User site
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The user site is written with Quarto in ``quarto/`` and is previewed with
``QUARTO_PYTHON="$PWD/.venv/bin/python" quarto preview quarto`` (live reload) or rendered
statically into ``quarto/_site/`` with ``QUARTO_PYTHON="$PWD/.venv/bin/python" quarto render quarto``.
Continuous integration never executes the notebooks of the site: it renders them from the cache
committed in ``quarto/_freeze/``. After changing a notebook, the site must therefore be rendered
locally and ``quarto/_freeze/`` committed together with the notebook; a test fails when they
disagree.

Citing
====================================

The first reference below is the one to cite for the methodology; the second is the work it extends.

.. code:: bibtex

    @article{Nava-Munoz2024,
      author  = {Nava-Mu{\~n}oz, Sergio and Graff, Mario and Escalante, Hugo Jair},
      title   = {Analysis of systems' performance in natural language processing competitions},
      journal = {Pattern Recognition Letters},
      year    = {2024},
      doi     = {10.1016/j.patrec.2024.03.010},
      url     = {https://doi.org/10.1016/j.patrec.2024.03.010}
    }

    @inproceedings{Nava-Munoz2023,
      author    = {Nava-Mu{\~n}oz, Sergio and Graff Guerrero, Mario and Escalante, Hugo Jair},
      title     = {Comparison of Classifiers in Challenge Scheme},
      booktitle = {Pattern Recognition},
      series    = {Lecture Notes in Computer Science},
      volume    = {13902},
      pages     = {89--98},
      year      = {2023},
      publisher = {Springer Nature Switzerland},
      doi       = {10.1007/978-3-031-33783-3_9},
      url       = {https://doi.org/10.1007/978-3-031-33783-3_9}
    }

References
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Berg-Kirkpatrick, T., Burkett, D. and Klein, D. (2012). An empirical investigation of
  statistical significance in NLP. *EMNLP-CoNLL 2012*.
- Chernick, M. R. and LaBudde, R. A. (2011). *An Introduction to Bootstrap Methods with
  Applications to R*. Wiley.
- Efron, B. and Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman and
  Hall/CRC.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian
  Journal of Statistics*, 6, 65-70.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and
  powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B*,
  57, 289-300.

API
====================================

.. toctree::
   :maxdepth: 1

   interface_api
   metrics_api
   measurements_api
   bootstrap_api
   performance_api
   utils_api
