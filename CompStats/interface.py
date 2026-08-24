# Copyright 2025 Sergio Nava Muñoz and Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
from CompStats.bootstrap import StatisticSamples
from CompStats.utils import progress_bar
from CompStats import measurements
from CompStats.measurements import SE, CI
from CompStats.utils import dataframe


class Perf(object):
    """Perf is an entry point to CompStats

    :param y_true: True measurement or could be a pandas.DataFrame where column label 'y' corresponds to the true measurement.
    :type y_true: numpy.ndarray or pandas.DataFrame
    :param func: Function (or list of functions) to measure the performance. Whether the best algorithm has the highest or the lowest value is given by :py:attr:`BiB` -- either the constructor's default, or, when a callable already carries its own :py:attr:`BiB` attribute (e.g. built by a :py:mod:`CompStats.metrics` wrapper's ``.measure`` factory), that tag takes precedence. A list of functions combines them into a single, multi-measure :py:class:`Perf.`
    :type func: Function, or list of functions, where the first argument is :math:`y` and the second is :math:`\\hat{y}.`
    :param BiB: Bigger is Better; the default direction used for any measure in :py:attr:`func` that doesn't already carry its own :py:attr:`BiB` attribute. A single bool applies to every measure; a list applies element-wise, one entry per measure in :py:attr:`func`.
    :type BiB: bool or list of bool
    :param measure_names: Display name for each measure, only relevant when more than one measure is given; defaults to each function's ``__name__``.
    :type measure_names: list
    :param y_pred: Predictions, the algorithms will be identified with alg-k where k=1 is the first argument included in :py:attr:`args.`
    :type y_pred: numpy.ndarray
    :param kwargs: Predictions, the algorithms will be identified using the keyword
    :type kwargs: numpy.ndarray
    :param n_jobs: Number of jobs to compute the statistic, default=-1 corresponding to use all threads.
    :type n_jobs: int
    :param num_samples: Number of bootstrap samples, default=500.
    :type num_samples: int
    :param use_tqdm: Whether to use tqdm.tqdm to visualize the progress, default=True.
    :type use_tqdm: bool


    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.base import clone
    >>> from CompStats.interface import Perf
    >>> X, y = load_iris(return_X_y=True)
    >>> _ = train_test_split(X, y, test_size=0.3)
    >>> X_train, X_val, y_train, y_val = _
    >>> m = LinearSVC().fit(X_train, y_train)
    >>> hy = m.predict(X_val)
    >>> perf = Perf(y_val, hy, name='LinearSVC')
    >>> ens = RandomForestClassifier().fit(X_train, y_train)
    >>> perf(ens.predict(X_val), name='forest')
    >>> perf
    <Perf>
    Statistic with its standard error (se)
    statistic (se)
    0.9792 (0.0221) <= LinearSVC
    0.9744 (0.0246) <= forest

    If an algorithm's prediction is missing, this can be included by calling the instance, as can be seen in the following instruction. Note that the algorithm's name can also be given with the keyword :py:attr:`name.`

    >>> lr = LogisticRegression().fit(X_train, y_train)
    >>> perf(lr.predict(X_val), name='Log. Reg.')
    <Perf>
    Statistic with its standard error (se)
    statistic (se)
    1.0000 (0.0000) <= Log. Reg.
    0.9792 (0.0221) <= alg-1
    0.9744 (0.0246) <= forest

    The performance function used to compare the algorithms can be changed, and the same bootstrap samples would be used if the instance were cloned. Consequently, the values are computed using the same samples, as can be seen in the following example.

    >>> perf_error = clone(perf)
    >>> perf_error.func = lambda y, hy: (y != hy).mean()
    >>> perf_error.BiB = False
    >>> perf_error
    <Perf>
    Statistic with its standard error (se)
    statistic (se)
    0.0000 (0.0000) <= Log. Reg.
    0.0222 (0.0237) <= alg-1
    0.0222 (0.0215) <= forest

    When several algorithms are compared, :py:meth:`difference`'s p-values can be
    adjusted for multiple comparisons by passing a
    :py:func:`statsmodels.stats.multitest.multipletests` method name (e.g.
    ``'bonferroni'``, ``'holm'``, ``'fdr_bh'``) as :py:attr:`correction` --
    both to :py:meth:`Difference.p_value` directly and, so that plots reflect
    the same adjusted significance, to :py:meth:`plot`/:py:meth:`dataframe`.

    >>> diff = perf.difference()
    >>> diff.p_value()
    {'alg-1': np.float64(0.3), 'forest': np.float64(0.2)}
    >>> diff.p_value(correction='bonferroni')
    {'alg-1': np.float64(0.6), 'forest': np.float64(0.4)}
    >>> perf.plot(correction='bonferroni')

    Two or more measures can be combined into a single :py:class:`Perf` instance
    (e.g. macro-F1 together with macro-recall) by passing a list of functions
    to :py:attr:`func` -- see :py:mod:`CompStats.metrics`'s
    ``.measure`` factories (e.g. :py:func:`~CompStats.metrics.f1_score.measure`). Every
    measure is evaluated on the same bootstrap resamples, so comparisons across
    algorithms remain paired for each measure.

    >>> from CompStats.metrics import f1_score, recall_score
    >>> perf = Perf(y_val, hy, forest=ens.predict(X_val),
    ...             func=[f1_score.measure(average='macro'),
    ...                   recall_score.measure(average='macro')])

    With multiple measures, :py:attr:`correction` is applied independently
    per measure (i.e. per column), so one metric's correction never mixes
    with another's.

    >>> diff = perf.difference()
    >>> diff.p_value()
    {'alg-1': array([1. , 0.3]), 'forest': array([0.2, 1. ])}
    >>> diff.p_value(correction='bonferroni')
    {'alg-1': array([1. , 0.6]), 'forest': array([0.4, 1. ])}
    """

    def __init__(self, y_true, *y_pred,
                 name: str = None,
                 func=balanced_accuracy_score,
                 BiB: bool = True,
                 measure_names: list = None,
                 num_samples: int = 500,
                 n_jobs: int = -1,
                 use_tqdm=True,
                 **kwargs):
        assert len(self._as_list(func)) >= 1
        self._func = func
        self._BiB = BiB
        self.measure_names = measure_names
        algs = {}
        if name is not None:
            if isinstance(name, str):
                name = [name]
        else:
            name = [f'alg-{k + 1}' for k, _ in enumerate(y_pred)]
        for key, v in zip(name, y_pred):
            algs[key] = np.asanyarray(v)
        algs.update(**kwargs)
        self.predictions = algs
        self.y_true = y_true
        self.num_samples = num_samples
        self.n_jobs = n_jobs
        self.use_tqdm = use_tqdm
        self.sorting_func = np.linalg.norm
        self._init()

    @staticmethod
    def _as_list(value):
        """Normalize a func argument into a list of callables"""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @property
    def _measures(self):
        """List of (callable, BiB) pairs, one per measure being evaluated

        Each callable's own :py:attr:`BiB` attribute (set by, e.g., a
        :py:meth:`metrics.py <CompStats.metrics>` wrapper's ``.measure`` factory)
        takes precedence over :py:attr:`BiB`, the constructor's default direction.
        """
        funcs = self._as_list(self.func)
        default = self.BiB
        if isinstance(default, (list, tuple, np.ndarray)):
            defaults = list(default)
        else:
            defaults = [default] * len(funcs)
        return [(f, bool(getattr(f, 'BiB', d)))
                for f, d in zip(funcs, defaults)]

    @property
    def _bib(self):
        """Scalar or per-measure array combining every measure's tagged BiB"""
        measures = self._measures
        if len(measures) == 1:
            return measures[0][1]
        return np.array([b for _, b in measures])

    def _init(self):
        """Compute the bootstrap statistic"""

        bib = self._bib
        if hasattr(self, '_statistic_samples'):
            _ = self.statistic_samples
            _.BiB = bib
        else:
            _ = StatisticSamples(statistic=self.statistic_func,
                                 n_jobs=self.n_jobs,
                                 num_samples=self.num_samples,
                                 BiB=bib)
            _.samples(N=self.y_true.shape[0])
        self.statistic_samples = _

    def get_params(self):
        """Parameters"""

        return dict(y_true=self.y_true,
                    func=self.func,
                    BiB=self.BiB,
                    measure_names=self._measure_names,
                    num_samples=self.num_samples,
                    n_jobs=self.n_jobs)

    def __sklearn_clone__(self):
        klass = self.__class__
        params = self.get_params()
        ins = klass(**params)
        ins.predictions = dict(self.predictions)
        ins._statistic_samples._samples = self.statistic_samples._samples
        ins.sorting_func = self.sorting_func
        return ins

    def __repr__(self):
        """Prediction statistics with standard error in parenthesis"""
        func_name = self.statistic_func.__name__
        statistic = self.statistic
        if isinstance(statistic, dict):
            return f"<{self.__class__.__name__}(func={func_name})>\n{self}"
        elif isinstance(statistic, float):
            return f"<{self.__class__.__name__}(func={func_name}, statistic={statistic:0.4f}, se={self.se:0.4f})>"
        desc = [f'{k:0.4f}' for k in statistic]
        desc = ', '.join(desc)
        desc_se = [f'{k:0.4f}' for k in self.se]
        desc_se = ', '.join(desc_se)
        return f"<{self.__class__.__name__}(func={func_name}, statistic=[{desc}], se=[{desc_se}])>"

    def __str__(self):
        """Prediction statistics with standard error in parenthesis"""
        if not isinstance(self.statistic, dict):
            return self.__repr__()

        se = self.se
        output = ["Statistic with its standard error (se)"]
        output.append("statistic (se)")
        for key, value in self.statistic.items():
            if isinstance(value, float):
                desc = f'{value:0.4f} ({se[key]:0.4f}) <= {key}'
            else:
                desc = [f'{v:0.4f} ({k:0.4f})'
                        for v, k in zip(value, se[key])]
                desc = ', '.join(desc)
                desc = f'{desc} <= {key}'
            output.append(desc)
        return "\n".join(output)

    def __call__(self, y_pred, name=None):
        """Add predictions"""
        if name is None:
            k = len(self.predictions) + 1
            if k == 0:
                k = 1
            name = f'alg-{k}'
        self.best = None
        self.statistic = None
        self.predictions[name] = np.asanyarray(y_pred)
        samples = self._statistic_samples
        calls = samples.calls
        if name in calls:
            del calls[name]
        return self

    def difference(self, wrt: str = None):
        """Compute the difference w.r.t any algorithm by default is the best

        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.base import clone
        >>> from CompStats.interface import Perf
        >>> X, y = load_iris(return_X_y=True)
        >>> _ = train_test_split(X, y, test_size=0.3)
        >>> X_train, X_val, y_train, y_val = _
        >>> m = LinearSVC().fit(X_train, y_train)
        >>> hy = m.predict(X_val)
        >>> ens = RandomForestClassifier().fit(X_train, y_train)
        >>> perf = Perf(y_val, hy, forest=ens.predict(X_val))
        >>> perf.difference()
        <Difference>
        difference p-values w.r.t alg-1
        forest 0.06
        """
        if wrt is None:
            wrt = self.best
        if isinstance(wrt, str):
            base = self.statistic_samples.calls[wrt]
        else:
            base = np.array([self.statistic_samples.calls[key][:, col]
                            for col, key in enumerate(wrt)]).T
        BiB = self.statistic_samples.BiB
        sign = np.where(BiB, 1, -1) if isinstance(BiB, np.ndarray) else (1 if BiB else -1)
        diff = dict()
        for k, v in self.statistic_samples.calls.items():
            if base.ndim == 1 and k == wrt:
                continue
            diff[k] = sign * (base - v)
        diff_ins = Difference(statistic_samples=clone(self.statistic_samples),
                              statistic=self.statistic)
        diff_ins.sorting_func = self.sorting_func
        diff_ins.statistic_samples.calls = diff
        diff_ins.statistic_samples.info['best'] = self.best
        diff_ins.best = self.best
        return diff_ins

    @property
    def best(self):
        """System with best performance"""
        if hasattr(self, '_best') and self._best is not None:
            return self._best
        if not isinstance(self.statistic, dict):
            key, value = list(self.statistic_samples.calls.items())[0]
            if value.ndim == 1:
                self._best = key
            else:
                self._best = np.array([key] * value.shape[1])
            return self._best
        BiB = self.statistic_samples.BiB
        keys = np.array(list(self.statistic.keys()))
        data = np.asanyarray([self.statistic[k]
                              for k in keys])
        if isinstance(self.statistic[keys[0]], np.ndarray):
            argmax_idx = data.argmax(axis=0)
            argmin_idx = data.argmin(axis=0)
            if isinstance(BiB, np.ndarray):
                best = np.where(BiB, argmax_idx, argmin_idx)
            else:
                best = argmax_idx if BiB else argmin_idx
        else:
            best = data.argmax() if bool(BiB) else data.argmin()
        self._best = keys[best]
        return self._best

    @best.setter
    def best(self, value):
        self._best = value

    @property
    def sorting_func(self):
        """Rank systems when multiple performances are used"""
        return self._sorting_func

    @sorting_func.setter
    def sorting_func(self, value):
        self._sorting_func = value

    @property
    def statistic(self):
        """Statistic

        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from CompStats.interface import Perf
        >>> X, y = load_iris(return_X_y=True)
        >>> _ = train_test_split(X, y, test_size=0.3)
        >>> X_train, X_val, y_train, y_val = _
        >>> m = LinearSVC().fit(X_train, y_train)
        >>> hy = m.predict(X_val)
        >>> ens = RandomForestClassifier().fit(X_train, y_train)
        >>> perf = Perf(y_val, hy, forest=ens.predict(X_val))
        >>> perf.statistic
        {'alg-1': 1.0, 'forest': 0.9500891265597148}
        """
        if hasattr(self, '_statistic') and self._statistic is not None:
            return self._statistic
        bib = self._bib
        BiB = bool(np.all(bib)) if isinstance(bib, np.ndarray) else bib
        data = sorted([(k, self.statistic_func(self.y_true, v))
                       for k, v in self.predictions.items()],
                      key=lambda x: self.sorting_func(x[1]),
                      reverse=BiB)
        if len(data) == 1:
            self._statistic = data[0][1]
        else:
            self._statistic = dict(data)
        return self._statistic

    @statistic.setter
    def statistic(self, value):
        """statistic setter"""
        self._statistic = value

    @property
    def se(self):
        """Standard Error

        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from CompStats.interface import Perf
        >>> X, y = load_iris(return_X_y=True)
        >>> _ = train_test_split(X, y, test_size=0.3)
        >>> X_train, X_val, y_train, y_val = _
        >>> m = LinearSVC().fit(X_train, y_train)
        >>> hy = m.predict(X_val)
        >>> ens = RandomForestClassifier().fit(X_train, y_train)
        >>> perf = Perf(y_val, hy, forest=ens.predict(X_val))
        >>> perf.se
        {'alg-1': 0.0, 'forest': 0.026945730782184187}
        """

        output = SE(self.statistic_samples)
        if len(output) == 1:
            return list(output.values())[0]
        return output

    @property
    def ci(self):
        """Confidence interval

        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from CompStats.interface import Perf
        >>> X, y = load_iris(return_X_y=True)
        >>> _ = train_test_split(X, y, test_size=0.3)
        >>> X_train, X_val, y_train, y_val = _
        >>> m = LinearSVC().fit(X_train, y_train)
        >>> hy = m.predict(X_val)
        >>> ens = RandomForestClassifier().fit(X_train, y_train)
        >>> perf = Perf(y_val, hy, name='LinearSVC')
        >>> perf.ci
        (np.float64(0.9333333333333332), np.float64(1.0))
        """

        output = CI(self.statistic_samples)
        if len(output) == 1:
            return list(output.values())[0]
        return output

    def plot(self, value_name: str = None,
             var_name: str = 'Performance',
             alg_legend: str = 'Algorithm',
             perf_names: list = None,
             CI: float = 0.05,
             kind: str = 'point', linestyle: str = 'none',
             col_wrap: int = 3, capsize: float = 0.2,
             comparison: bool = True,
             right: bool = True,
             correction: str = None,
             comp_legend: str = 'Comparison',
             winner_legend: str = 'Best',
             tie_legend: str = 'Equivalent',
             loser_legend: str = 'Different',
             palette: object = None,
             **kwargs):
        """plot with seaborn

        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from CompStats.interface import Perf
        >>> X, y = load_iris(return_X_y=True)
        >>> _ = train_test_split(X, y, test_size=0.3)
        >>> X_train, X_val, y_train, y_val = _
        >>> m = LinearSVC().fit(X_train, y_train)
        >>> hy = m.predict(X_val)
        >>> ens = RandomForestClassifier().fit(X_train, y_train)
        >>> perf = Perf(y_val, hy,
                        func=lambda y, hy: (y != hy).mean(), BiB=False,
                        forest=ens.predict(X_val))
        >>> perf.plot()
        """
        import seaborn as sns
        if value_name is None:
            measures = self._measures
            if len(measures) > 1:
                value_name = 'Value'
            elif measures[0][1]:
                value_name = 'Score'
            else:
                value_name = 'Error'
        if not isinstance(self.statistic, dict):
            comparison = False
        best = self.best
        if isinstance(best, np.ndarray):
            if best.shape[0] < col_wrap:
                col_wrap = best.shape[0]
        df = self.dataframe(value_name=value_name, var_name=var_name,
                            alg_legend=alg_legend, perf_names=perf_names,
                            comparison=comparison, alpha=CI, right=right,
                            correction=correction,
                            comp_legend=comp_legend,
                            winner_legend=winner_legend,
                            tie_legend=tie_legend,
                            loser_legend=loser_legend)
        if var_name not in df.columns:
            var_name = None
            col_wrap = None
        ci = lambda x: measurements.CI(x, alpha=CI)
        if comparison:
            kwargs.update(dict(hue=comp_legend))
            if palette is None:
                pal = sns.color_palette("Paired")
                palette = {winner_legend: pal[1],
                           tie_legend: pal[3],
                           loser_legend: pal[5]}
        f_grid = sns.catplot(df, x=value_name, errorbar=ci,
                             y=alg_legend, col=var_name,
                             kind=kind, linestyle=linestyle,
                             col_wrap=col_wrap, capsize=capsize,
                             palette=palette,
                             **kwargs)
        return f_grid

    def dataframe(self, comparison: bool = False,
                  right: bool = True,
                  alpha: float = 0.05,
                  correction: str = None,
                  value_name: str = 'Score',
                  var_name: str = 'Performance',
                  alg_legend: str = 'Algorithm',
                  comp_legend: str = 'Comparison',
                  winner_legend: str = 'Best',
                  tie_legend: str = 'Equivalent',
                  loser_legend: str = 'Different',
                  perf_names: str = None):
        """Dataframe

        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from CompStats.interface import Perf
        >>> X, y = load_iris(return_X_y=True)
        >>> _ = train_test_split(X, y, test_size=0.3)
        >>> X_train, X_val, y_train, y_val = _
        >>> m = LinearSVC().fit(X_train, y_train)
        >>> hy = m.predict(X_val)
        >>> ens = RandomForestClassifier().fit(X_train, y_train)
        >>> perf = Perf(y_val, hy, forest=ens.predict(X_val))
        >>> df = perf.dataframe()
        """
        if perf_names is None and isinstance(self.best, np.ndarray):
            perf_names = self.measure_names
            if perf_names is None:
                func_name = self.statistic_func.__name__
                perf_names = [f'{func_name}({i})'
                              for i, k in enumerate(self.best)]
        df = dataframe(self, value_name=value_name,
                       var_name=var_name,
                       alg_legend=alg_legend,
                       perf_names=perf_names)
        if not comparison:
            return df
        df[comp_legend] = tie_legend
        diff = self.difference()
        best = self.best
        if isinstance(best, str):
            for name, p in diff.p_value(right=right, correction=correction).items():
                if p >= alpha:
                    continue
                df.loc[df[alg_legend] == name, comp_legend] = loser_legend
            df.loc[df[alg_legend] == best, comp_legend] = winner_legend
        else:
            p_values = diff.p_value(right=right, correction=correction)
            systems = list(p_values.keys())
            p_values = np.array([p_values[k] for k in systems])
            for name, p_value, winner in zip(perf_names,
                                             p_values.T,
                                             best):
                mask = df[var_name] == name
                for alg, p in zip(systems, p_value):
                    if p >= alpha and winner != alg:
                        continue
                    _ = mask & (df[alg_legend] == alg)
                    if winner == alg:
                        df.loc[_, comp_legend] = winner_legend
                    else:
                        df.loc[_, comp_legend] = loser_legend
        return df

    @property
    def n_jobs(self):
        """Number of jobs to compute the statistics"""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def statistic_func(self):
        """Statistic function

        A single :py:attr:`func` callable is returned as-is; when more than
        one measure is given (a list passed to :py:attr:`func`), a composite
        callable is returned that concatenates every measure's output into a
        single vector, evaluated on the same bootstrap samples.
        """
        measures = self._measures
        if len(measures) == 1:
            return measures[0][0]
        funcs = [f for f, _ in measures]
        names = self.measure_names

        def composite(y, hy):
            return np.concatenate([np.atleast_1d(f(y, hy)) for f in funcs])
        composite.__name__ = '+'.join(names)
        return composite

    @property
    def measure_names(self):
        """Display name for each measure, used when combining more than one

        Defaults to each measure's function ``__name__`` (available because
        every :py:mod:`CompStats.metrics` wrapper's inner function is
        ``functools.wraps``-decorated with the corresponding sklearn metric).
        """
        if self._measure_names is not None:
            return self._measure_names
        measures = self._measures
        if len(measures) <= 1:
            return None
        return [getattr(f, '__name__', f'measure-{i}')
                for i, (f, _) in enumerate(measures)]

    @measure_names.setter
    def measure_names(self, value):
        self._measure_names = value

    @property
    def statistic_samples(self):
        """Statistic Samples"""

        samples = self._statistic_samples
        algs = set(samples.calls.keys())
        algs = set(self.predictions.keys()) - algs
        if len(algs):
            for key in progress_bar(algs, use_tqdm=self.use_tqdm):
                samples(self.y_true, self.predictions[key], name=key)
        return self._statistic_samples

    @statistic_samples.setter
    def statistic_samples(self, value):
        self._statistic_samples = value

    @property
    def num_samples(self):
        """Number of bootstrap samples"""
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value):
        self._num_samples = value

    @property
    def predictions(self):
        """Predictions"""
        return self._predictions

    @predictions.setter
    def predictions(self, value):
        self._predictions = value

    @property
    def y_true(self):
        """True output, gold standard o :math:`y`"""

        return self._y_true

    @y_true.setter
    def y_true(self, value):
        if isinstance(value, pd.DataFrame):
            self._y_true = value['y'].to_numpy()
            algs = {}
            for c in value.columns:
                if c == 'y':
                    continue
                algs[c] = value[c].to_numpy()
            self.predictions.update(algs)
            return
        self._y_true = np.asanyarray(value)

    @property
    def func(self):
        """Function (or list of functions) used to measure the performance"""
        return self._func

    @func.setter
    def func(self, value):
        self._func = value
        if hasattr(self, '_statistic_samples'):
            self._statistic_samples.statistic = self.statistic_func
            self._statistic_samples.BiB = self._bib

    @property
    def BiB(self):
        """Bigger is Better; default direction for measures without their own :py:attr:`BiB` tag"""
        return self._BiB

    @BiB.setter
    def BiB(self, value):
        self._BiB = value
        if hasattr(self, '_statistic_samples'):
            self._statistic_samples.BiB = self._bib


@dataclass
class Difference:
    """Difference

    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.base import clone
    >>> from CompStats.interface import Perf
    >>> X, y = load_iris(return_X_y=True)
    >>> _ = train_test_split(X, y, test_size=0.3)
    >>> X_train, X_val, y_train, y_val = _
    >>> m = LinearSVC().fit(X_train, y_train)
    >>> hy = m.predict(X_val)
    >>> ens = RandomForestClassifier().fit(X_train, y_train)
    >>> perf = Perf(y_val, hy, forest=ens.predict(X_val))
    >>> diff = perf.difference()
    >>> diff
    <Difference>
    difference p-values w.r.t alg-1
    0.0780 <= forest
    """

    statistic_samples: StatisticSamples = None
    statistic: dict = None
    best: str = None

    @property
    def sorting_func(self):
        """Rank systems when multiple performances are used"""
        return self._sorting_func

    @sorting_func.setter
    def sorting_func(self, value):
        self._sorting_func = value

    def __repr__(self):
        """p-value"""
        return f"<{self.__class__.__name__}>\n{self}"

    def __str__(self):
        """p-value"""
        if isinstance(self.best, str):
            best = f' w.r.t {self.best}'
        else:
            best = ''
        output = [f"difference p-values {best}"]
        best = self.best
        if isinstance(best, np.ndarray):
            desc = ', '.join(best)
            output.append(f'{desc} <= Best')
        for key, value in self.p_value().items():
            if isinstance(value, float):
                output.append(f'{value:0.4f} <= {key}')
            else:
                desc = [f'{v:0.4f}' for v in value]
                desc = ', '.join(desc)
                desc = f'{desc} <= {key}'
                output.append(desc)
        return "\n".join(output)

    def _delta_best(self):
        """Compute multiple delta"""
        if isinstance(self.best, str):
            return self.statistic[self.best]
        keys = np.unique(self.best)
        statistic = np.array([self.statistic[k]
                              for k in keys])
        m = {v: k for k, v in enumerate(keys)}
        best = np.array([m[x] for x in self.best])
        return statistic[best, np.arange(best.shape[0])]

    def p_value(self, right: bool = True, correction: str = None):
        """Compute p_value of the differences

        :param right: Estimate the p-value using :math:`\\text{sample} \\geq 2\\delta`
        :type right: bool
        :param correction: Method to adjust for multiple comparisons, passed to :py:func:`statsmodels.stats.multitest.multipletests` (e.g. ``'bonferroni'``, ``'holm'``, ``'fdr_bh'``); ``None`` (default) leaves the p-values uncorrected. With a single measure, the family of comparisons is every other system against :py:attr:`best`; with multiple measures, each measure is corrected as its own family (i.e. per column) so metrics do not contaminate each other's correction.
        :type correction: str

        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.naive_bayes import GaussianNB
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.base import clone
        >>> from CompStats.interface import Perf
        >>> X, y = load_iris(return_X_y=True)
        >>> _ = train_test_split(X, y, test_size=0.3)
        >>> X_train, X_val, y_train, y_val = _
        >>> m = LinearSVC().fit(X_train, y_train)
        >>> hy = m.predict(X_val)
        >>> perf = Perf(y_val, hy, name='LinearSVC')
        >>> ens = RandomForestClassifier().fit(X_train, y_train)
        >>> perf(ens.predict(X_val), name='forest')
        >>> nb = GaussianNB().fit(X_train, y_train)
        >>> perf(nb.predict(X_val), name='bayes')
        >>> diff = perf.difference()
        >>> diff.p_value()
        {'forest': np.float64(0.3), 'bayes': np.float64(0.2)}
        >>> diff.p_value(correction='bonferroni')
        {'forest': np.float64(0.6), 'bayes': np.float64(0.4)}
        """
        values = []
        BiB = self.statistic_samples.BiB
        sign = np.where(BiB, 1, -1) if isinstance(BiB, np.ndarray) else (1 if BiB else -1)
        delta_best = self._delta_best()
        for k, v in self.statistic_samples.calls.items():
            delta = 2 * sign * (delta_best - self.statistic[k])
            if not isinstance(delta_best, np.ndarray):
                if right:
                    values.append((k, (v >= delta).mean()))
                else:
                    values.append((k, (v <= 0).mean()))
            else:
                if right:
                    values.append((k, (v >= delta).mean(axis=0)))
                else:
                    values.append((k, (v <= 0).mean(axis=0)))
        values.sort(key=lambda x: self.sorting_func(x[1]))
        if correction is None:
            return dict(values)
        keys = [k for k, _ in values]
        raw = np.array([v for _, v in values])
        if raw.ndim == 1:
            corrected = multipletests(raw, method=correction)[1]
        else:
            corrected = np.column_stack(
                [multipletests(raw[:, col], method=correction)[1]
                 for col in range(raw.shape[1])])
        return dict(zip(keys, corrected))

    def dataframe(self, value_name: str = 'Score',
                  var_name: str = 'Best',
                  alg_legend: str = 'Algorithm',
                  sig_legend: str = 'Significant',
                  perf_names: str = None,
                  right: bool = True,
                  alpha: float = 0.05,
                  correction: str = None):
        """Dataframe"""
        if perf_names is None and isinstance(self.best, np.ndarray):
            perf_names = [f'{alg}({k})'
                          for k, alg in enumerate(self.best)]
        df = dataframe(self, value_name=value_name,
                       var_name=var_name,
                       alg_legend=alg_legend,
                       perf_names=perf_names)
        df[sig_legend] = False
        if isinstance(self.best, str):
            for name, p in self.p_value(right=right, correction=correction).items():
                if p >= alpha:
                    continue
                df.loc[df[alg_legend] == name, sig_legend] = True
        else:
            p_values = self.p_value(right=right, correction=correction)
            systems = list(p_values.keys())
            p_values = np.array([p_values[k] for k in systems])
            for name, p_value in zip(perf_names, p_values.T):
                mask = df[var_name] == name
                for alg, p in zip(systems, p_value):
                    if p >= alpha:
                        continue
                    _ = mask & (df[alg_legend] == alg)
                    df.loc[_, sig_legend] = True
        return df

    def plot(self, value_name: str = 'Difference',
             var_name: str = 'Best',
             alg_legend: str = 'Algorithm',
             sig_legend: str = 'Significant',
             perf_names: list = None,
             alpha: float = 0.05,
             right: bool = True,
             correction: str = None,
             kind: str = 'point', linestyle: str = 'none',
             col_wrap: int = 3, capsize: float = 0.2,
             set_refline: bool = True,
             **kwargs):
        """Plot

        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.base import clone
        >>> from CompStats.interface import Perf
        >>> X, y = load_iris(return_X_y=True)
        >>> _ = train_test_split(X, y, test_size=0.3)
        >>> X_train, X_val, y_train, y_val = _
        >>> m = LinearSVC().fit(X_train, y_train)
        >>> hy = m.predict(X_val)
        >>> ens = RandomForestClassifier().fit(X_train, y_train)
        >>> perf = Perf(y_val, hy, forest=ens.predict(X_val))
        >>> diff = perf.difference()
        >>> diff.plot()
        """
        import seaborn as sns
        df = self.dataframe(value_name=value_name,
                            var_name=var_name,
                            alg_legend=alg_legend,
                            sig_legend=sig_legend,
                            perf_names=perf_names,
                            alpha=alpha, right=right,
                            correction=correction)
        title = var_name
        if var_name not in df.columns:
            var_name = None
            col_wrap = None
        ci = lambda x: measurements.CI(x, alpha=2 * alpha)
        f_grid = sns.catplot(df, x=value_name, errorbar=ci,
                             y=alg_legend, col=var_name,
                             kind=kind, linestyle=linestyle,
                             col_wrap=col_wrap, capsize=capsize,
                             hue=sig_legend,
                             **kwargs)
        if set_refline:
            f_grid.refline(x=0)
        if isinstance(self.best, str):
            f_grid.facet_axis(0, 0).set_title(f'{title} = {self.best}')
        return f_grid
