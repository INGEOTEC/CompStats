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
import pandas as pd
import numpy as np
from CompStats.bootstrap import StatisticSamples
from CompStats.utils import progress_bar
from CompStats import measurements
from CompStats.measurements import SE
from CompStats.utils import dataframe


class Perf(object):
    """Perf is an entry point to CompStats

    :param y_true: True measurement or could be a pandas.DataFrame where column label 'y' corresponds to the true measurement.
    :type y_true: numpy.ndarray or pandas.DataFrame
    :param score_func: Function to measure the performance, it is assumed that the best algorithm has the highest value.
    :type score_func: Function where the first argument is :math:`y` and the second is :math:`\\hat{y}.`
    :param error_func: Function to measure the performance where the best algorithm has the lowest value.
    :type error_func: Function where the first argument is :math:`y` and the second is :math:`\\hat{y}.` 
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
    >>> ens = RandomForestClassifier().fit(X_train, y_train)
    >>> perf = Perf(y_val, hy, forest=ens.predict(X_val))
    >>> perf
    <Perf>
    Statistic with its standard error (se)
    statistic (se)
    0.9792 (0.0221) <= alg-1
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
    >>> perf_error.error_func = lambda y, hy: (y != hy).mean()
    >>> perf_error
    <Perf>
    Statistic with its standard error (se)
    statistic (se)
    0.0000 (0.0000) <= Log. Reg.
    0.0222 (0.0237) <= alg-1
    0.0222 (0.0215) <= forest

    """
    def __init__(self, y_true, *y_pred,
                 score_func=balanced_accuracy_score,
                 error_func=None,
                 num_samples: int=500,
                 n_jobs: int=-1,
                 use_tqdm=True,
                 **kwargs):
        assert (score_func is None) ^ (error_func is None)
        self.score_func = score_func
        self.error_func = error_func
        algs = {}
        for k, v in enumerate(y_pred):
            algs[f'alg-{k+1}'] = np.asanyarray(v)
        algs.update(**kwargs)
        self.predictions = algs
        self.y_true = y_true
        self.num_samples = num_samples
        self.n_jobs = n_jobs
        self.use_tqdm = use_tqdm
        self.sorting_func = np.linalg.norm
        self._init()

    def _init(self):
        """Compute the bootstrap statistic"""

        bib = True if self.score_func is not None else False
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
                    score_func=self.score_func,
                    error_func=self.error_func,
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
        arg = 'score_func' if self.error_func is None else 'error_func'
        func_name = self.statistic_func.__name__
        statistic = self.statistic
        if isinstance(statistic, dict):
            return f"<{self.__class__.__name__}({arg}={func_name})>\n{self}"
        elif isinstance(statistic, float):
            return f"<{self.__class__.__name__}({arg}={func_name}, statistic={statistic:0.4f}, se={self.se:0.4f})>"
        desc = [f'{k:0.4f}' for k in statistic]
        desc = ', '.join(desc)
        desc_se = [f'{k:0.4f}' for k in self.se]
        desc_se = ', '.join(desc_se)
        return f"<{self.__class__.__name__}({arg}={func_name}, statistic=[{desc}], se=[{desc_se}])>"

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
        self.predictions[name] = np.asanyarray(y_pred)
        samples = self._statistic_samples
        calls = samples.calls
        if name in calls:
            del calls[name]
        return self

    def difference(self, wrt: str=None):
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
        sign = 1 if self.statistic_samples.BiB else -1
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
        BiB = bool(self.statistic_samples.BiB)
        keys = np.array(list(self.statistic.keys()))
        data = np.asanyarray([self.statistic[k]
                              for k in keys])        
        if isinstance(self.statistic[keys[0]], np.ndarray):
            if BiB:
                best = data.argmax(axis=0)
            else:
                best = data.argmin(axis=0)
        else:
            if BiB:
                best = data.argmax()
            else:
                best = data.argmin()
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

        data = sorted([(k, self.statistic_func(self.y_true, v))
                       for k, v in self.predictions.items()],
                      key=lambda x: self.sorting_func(x[1]), 
                      reverse=self.statistic_samples.BiB)
        if len(data) == 1:
            return data[0][1]
        return dict(data)

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

    def plot(self, value_name:str=None,
             var_name:str='Performance',
             alg_legend:str='Algorithm',
             perf_names:list=None,
             CI:float=0.05,
             kind:str='point', linestyle:str='none',
             col_wrap:int=3, capsize:float=0.2,
             comparison:bool=True,
             right:bool=True,
             comp_legend:str='Comparison',
             winner_legend:str='Best',
             tie_legend:str='Equivalent',
             loser_legend:str='Different',
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
        >>> perf = Perf(y_val, hy, score_func=None,
                        error_func=lambda y, hy: (y != hy).mean(),
                        forest=ens.predict(X_val))
        >>> perf.plot()
        """
        import seaborn as sns
        if value_name is None:
            if self.score_func is not None:
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
        f_grid = sns.catplot(df, x=value_name, errorbar=ci,
                             y=alg_legend, col=var_name,
                             kind=kind, linestyle=linestyle,
                             col_wrap=col_wrap, capsize=capsize, **kwargs)
        return f_grid

    def dataframe(self, comparison:bool=False,
                  right:bool=True,
                  alpha:float=0.05,
                  value_name:str='Score',
                  var_name:str='Performance',
                  alg_legend:str='Algorithm',
                  comp_legend:str='Comparison',
                  winner_legend:str='Best',
                  tie_legend:str='Equivalent',
                  loser_legend:str='Different',
                  perf_names:str=None):
        """Dataframe"""
        if perf_names is None and isinstance(self.best, np.ndarray):
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
            for name, p in diff.p_value(right=right).items():
                if p >= alpha:
                    continue
                df.loc[df[alg_legend] == name, comp_legend] = loser_legend
            df.loc[df[alg_legend] == best, comp_legend] = winner_legend
        else:
            p_values = diff.p_value(right=right)
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
        """Statistic function"""
        if self.score_func is not None:
            return self.score_func
        return self.error_func

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
        self._y_true = value

    @property
    def score_func(self):
        """Score function"""
        return self._score_func

    @score_func.setter
    def score_func(self, value):
        self._score_func = value
        if value is not None:
            self.error_func = None
            if hasattr(self, '_statistic_samples'):
                self._statistic_samples.statistic = value
                self._statistic_samples.BiB = True

    @property
    def error_func(self):
        """Error function"""
        return self._error_func

    @error_func.setter
    def error_func(self, value):
        self._error_func = value
        if value is not None:
            self.score_func = None
            if hasattr(self, '_statistic_samples'):
                self._statistic_samples.statistic = value
                self._statistic_samples.BiB = False


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

    statistic_samples:StatisticSamples=None
    statistic:dict=None
    best:str=None

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

    def p_value(self, right:bool=True):
        """Compute p_value of the differences

        :param right: Estimate the p-value using :math:`\\text{sample} \\geq 2\\delta`
        :type right: bool  
        
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
        >>> diff.p_value()
        {'forest': np.float64(0.3)}
        """
        values = []
        sign = 1 if self.statistic_samples.BiB else -1
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
        return dict(values)

    def dataframe(self, value_name:str='Score',
                  var_name:str='Best',
                  alg_legend:str='Algorithm',
                  sig_legend:str='Significant',
                  perf_names:str=None,
                  right:bool=True,
                  alpha:float=0.05):
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
            for name, p in self.p_value(right=right).items():
                if p >= alpha:
                    continue
                df.loc[df[alg_legend] == name, sig_legend] = True
        else:
            p_values = self.p_value(right=right)
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

    def plot(self, value_name:str='Difference',
             var_name:str='Best',
             alg_legend:str='Algorithm',
             sig_legend:str='Significant',
             perf_names:list=None,
             alpha:float=0.05,
             right:bool=True,
             kind:str='point', linestyle:str='none',
             col_wrap:int=3, capsize:float=0.2,
             set_refline:bool=True,
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
                            alpha=alpha, right=right)
        title = var_name         
        if var_name not in df.columns:
            var_name = None
            col_wrap = None
        ci = lambda x: measurements.CI(x, alpha=2*alpha)
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
