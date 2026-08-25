.. _CompStats:

====================================
CompStats
====================================
.. image:: https://img.shields.io/badge/GitHub-CompStats-black?logo=github
        :target: https://github.com/INGEOTEC/CompStats

.. image:: https://github.com/INGEOTEC/CompStats/actions/workflows/test.yaml/badge.svg
		:target: https://github.com/INGEOTEC/CompStats/actions/workflows/test.yaml

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

.. image:: https://readthedocs.org/projects/compstats/badge/?version=latest
		:target: https://compstats.readthedocs.io/en/latest/?badge=latest

.. image:: https://colab.research.google.com/assets/colab-badge.svg
		:target: https://colab.research.google.com/github/INGEOTEC/CompStats/blob/docs/docs/CompStats.ipynb


Collaborative competitions have gained popularity in the scientific and technological fields. These competitions involve defining tasks, selecting evaluation scores, and devising result verification methods. In the standard scenario, participants receive a training set and are expected to provide a solution for a held-out dataset kept by organizers. An essential challenge for organizers arises when comparing algorithms' performance, assessing multiple participants, and ranking them. Statistical tools are often used for this purpose; however, traditional statistical methods often fail to capture decisive differences between systems' performance. :py:class:`CompStats` implements an evaluation methodology for statistically analyzing competition results and competition. :py:class:`CompStats` offers several advantages, including off-the-shell comparisons with correction mechanisms and the inclusion of confidence intervals.

.. note::

   This page is the **development documentation** for :py:class:`CompStats`: the internal
   architecture and the full API reference, aimed at people extending or contributing to the
   package. If you are looking to learn how to *use* CompStats -- installation, and tutorials for
   scikit-learn users, competition organizers, and correcting for multiple comparisons -- visit the
   `CompStats website <https://ingeotec.github.io/CompStats/>`_ instead.

Architecture
====================================

:py:class:`CompStats` is organized around one core data flow: raw predictions from one or more
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
  systems' predictions against shared ground truth, holding one :py:class:`StatisticSamples`
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

Citing
==========

If you find CompStats useful for any academic/scientific purpose, we would appreciate citations to the following reference:
  
.. code:: bibtex

    @article{Nava:2023,
    title = {{Comparison of Classifiers in Challenge Scheme}},
    year = {2023},
    journal = {Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)},
    author = {Nava-Mu{\~{n}}oz, Sergio and Graff Guerrero, Mario and Escalante, Hugo Jair},
    pages = {89--98},
    volume = {13902 LNCS},
    publisher = {Springer Science and Business Media Deutschland GmbH},
    url = {https://link.springer.com/chapter/10.1007/978-3-031-33783-3_9},
    isbn = {9783031337826},
    doi = {10.1007/978-3-031-33783-3{\_}9/COVER},
    issn = {16113349},
    keywords = {Bootstrap, Challenges, Performance}
    }


API
====================================

.. toctree::
   :maxdepth: 1

   metrics_api
   interface_api
   performance_api
   measurements_api
   bootstrap_api