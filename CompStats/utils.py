# Copyright 2024 Sergio Nava Muñoz and Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import wraps
try:
    USE_TQDM = True
    from tqdm import tqdm
except ImportError:
    USE_TQDM = False


def progress_bar(arg, use_tqdm: bool = True, **kwargs):
    """Wrap `arg` in a :py:class:`tqdm.tqdm` progress bar.

    Returns `arg` unchanged when tqdm is not installed or :py:attr:`use_tqdm` is
    `False`, so callers can iterate over the result the same way regardless of
    whether a progress bar is actually shown.

    :param arg: Iterable to wrap.
    :param use_tqdm: Whether to show the progress bar, default=True.
    :type use_tqdm: bool
    :param kwargs: Extra keyword arguments passed to :py:class:`tqdm.tqdm`.
    :return: `arg`, optionally wrapped in a :py:class:`tqdm.tqdm` iterator.
    """
    if not USE_TQDM or not use_tqdm:
        return arg
    return tqdm(arg, **kwargs)


def metrics_docs(hy_name='y_pred', bib: bool = True):
    """Decorator that injects the shared :py:class:`~CompStats.interface.Perf`
    docstring into a :py:mod:`CompStats.metrics` wrapper (e.g.
    :py:func:`~CompStats.metrics.f1_score`).

    :param hy_name: Name used for the predictions parameter in the generated
        docstring (e.g. ``y_pred`` or ``y_score``, matching the wrapped
        :py:mod:`sklearn.metrics` function's own parameter name).
    :type hy_name: str
    :param bib: Whether the wrapped function's measure is score-type (bigger
        is better) or error-type (smaller is better); used only to describe
        the measure's direction in the generated docstring, matching the
        ``.BiB`` tag set on the wrapper's ``.measure`` factory.
    :type bib: bool
    """

    def perf_docs(func):
        """Decorator to Perf to write :py:class:`~sklearn.metrics` documentation"""

        direction = 'score-type (bigger is better)' if bib else 'error-type (smaller is better)'
        func.__doc__ = f""":py:class:`~CompStats.interface.Perf` with :py:func:`~sklearn.metrics.{func.__name__}` as a {direction} :py:attr:`func.` The parameters not described can be found in :py:func:`~sklearn.metrics.{func.__name__}`.

    :param y_true: True measurement or could be a pandas.DataFrame where column label 'y' corresponds to the true measurement.
    :type y_true: numpy.ndarray or pandas.DataFrame
    :param {hy_name}: Predictions, the algorithms will be identified with alg-k where k=1 is the first argument included in :py:attr:`y_pred.`
    :type {hy_name}: numpy.ndarray
    :param kwargs: Predictions, the algorithms will be identified using the keyword
    :type kwargs: numpy.ndarray
    :param num_samples: Number of bootstrap samples, default=500.
    :type num_samples: int
    :param n_jobs: Number of jobs to compute the statistic, default=-1 corresponding to use all threads.
    :type n_jobs: int
    :param use_tqdm: Whether to use tqdm.tqdm to visualize the progress, default=True
    :type use_tqdm: bool

    :py:func:`~CompStats.metrics.{func.__name__}.measure` builds the tagged callable used internally as :py:attr:`func`; call it directly (e.g. ``{func.__name__}.measure(...)``) to combine this metric with others into a single, multi-measure :py:class:`~CompStats.interface.Perf` -- see :py:class:`~CompStats.interface.Perf`'s class docstring for a worked example.

    """ + func.__doc__

        @wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner
    return perf_docs


def dataframe(instance, value_name: str = 'Score',
              var_name: str = 'Performance',
              alg_legend: str = 'Algorithm',
              perf_names: list = None):
    """Melt a :py:class:`~CompStats.interface.Perf` or
    :py:class:`~CompStats.interface.Difference` instance's bootstrap samples into
    a long-format :py:class:`pandas.DataFrame`, ready for seaborn's ``catplot``
    (used by :py:meth:`~CompStats.interface.Perf.plot` and
    :py:meth:`~CompStats.interface.Difference.plot`).

    :param instance: Instance holding the bootstrap samples to melt.
    :type instance: CompStats.interface.Perf or CompStats.interface.Difference
    :param value_name: Column name for the statistic's value.
    :type value_name: str
    :param var_name: Column name identifying which measure a row belongs to,
        only used when `instance` holds more than one measure.
    :type var_name: str
    :param alg_legend: Column name identifying which algorithm a row belongs to.
    :type alg_legend: str
    :param perf_names: Display name for each measure, only used when `instance`
        holds more than one measure.
    :type perf_names: list
    :return: Long-format dataframe with one row per bootstrap sample.
    :rtype: pandas.DataFrame
    """
    import pandas as pd
    statistic = instance.statistic
    if not isinstance(statistic, dict):
        iter = instance.statistic_samples.keys()
    else:
        iter = statistic
    if isinstance(instance.best, str):
        calls = instance.statistic_samples.calls
        df = pd.DataFrame({k: calls[k]
                           for k in iter if k in calls})
        return df.melt(var_name=alg_legend,
                       value_name=value_name)
    df = pd.DataFrame()
    for key in iter:
        data = instance.statistic_samples[key]
        _df = pd.DataFrame(data,
                           columns=perf_names).melt(value_name=value_name,
                                                    var_name=var_name)
        _df[alg_legend] = key
        df = pd.concat((df, _df))
    return df
