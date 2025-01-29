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
from sklearn.metrics import f1_score
from sklearn.base import clone
from CompStats.bootstrap import StatisticSamples
from CompStats.utils import progress_bar
from CompStats.measurements import SE
from CompStats.performance import plot_performance, plot_difference


def macro(func):
    """Macro score"""

    def inner(y, hy):
        return func(y, hy, average='macro')
    return inner


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
    forest 0.3
    """

    statistic_samples:StatisticSamples=None
    best:str=None
    statistic:dict=None

    def __repr__(self):
        """p-value"""
        return f"<{self.__class__.__name__}>\n{self}"    

    def __str__(self):
        """p-value"""
        output = [f"difference p-values w.r.t {self.best}"]
        for k, v in self.p_value().items():
            output.append(f'{k} {v}')
        return "\n".join(output)

    def p_value(self):
        """Compute p_value of the differences
        
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
        for k, v in self.statistic_samples.calls.items():
            delta = 2 * sign * (self.statistic[self.best] - self.statistic[k])
            values.append((k, (v > delta).mean()))
        values.sort(key=lambda x: x[1])
        return dict(values)

    def plot(self, **kwargs):
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

        return plot_difference(self.statistic_samples, **kwargs)



class Perf(object):
    """Perf is an entry point to CompStats

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
    >>> perf
    <Perf>
    Prediction statistics with standard error
    alg-1 = 1.000 (0.000)
    forest = 0.978 (0.019)

    >>> perf_error = clone(perf)
    >>> perf_error.error_func = lambda y, hy: (y != hy).mean()
    >>> perf_error
    <Perf>
    Prediction statistics with standard error
    alg-1 = 0.000 (0.000)
    forest = 0.022 (0.018)    

    """
    def __init__(self, gold, *args,
                 score_func=macro(f1_score),
                 error_func=None,
                 num_samples: int=50,
                 n_jobs: int=-1,
                 **kwargs):
        assert (score_func is None) ^ (error_func is None)
        self.score_func = score_func
        self.error_func = error_func
        self.gold = gold
        algs = {}
        for k, v in enumerate(args):
            algs[f'alg-{k+1}'] = v
        algs.update(**kwargs)
        self.predictions = algs
        self.num_samples = num_samples
        self.n_jobs = n_jobs
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
            _.samples(N=self.gold.shape[0])
        self.statistic_samples = _

    def get_params(self):
        """Parameters"""

        return dict(gold=self.gold,
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
        return ins
    
    def __repr__(self):
        """Prediction statistics with standard error in parenthesis"""
        return f"<{self.__class__.__name__}>\n{self}"

    def __str__(self):
        """Prediction statistics with standard error in parenthesis"""

        se = self.se()
        output = ["Prediction statistics with standard error"]
        for key, value in self.statistic().items():
            output.append(f'{key} = {value:0.3f} ({se[key]:0.3f})')
        return "\n".join(output)

    def difference(self, wrt_to: str=None):
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
        if wrt_to is None:
            wrt_to = self.best[0]
        base = self.statistic_samples.calls[wrt_to]
        sign = 1 if self.statistic_samples.BiB else -1
        diff = dict()
        for k, v in self.statistic_samples.calls.items():
            if k == wrt_to:
                continue
            diff[k] = sign * (base - v)
        diff_ins = Difference(statistic_samples=clone(self.statistic_samples),
                              statistic=self.statistic(),
                              best=self.best[0])
        diff_ins.statistic_samples.calls = diff
        diff_ins.statistic_samples.info['best'] = self.best[0]
        return diff_ins

    @property
    def best(self):
        """Best system"""

        try:
            return self._best
        except AttributeError:
            statistic = [(k, v) for k, v in self.statistic().items()]
            statistic = sorted(statistic, key=lambda x: x[1],
                               reverse=self.statistic_samples.BiB)
            self._best = statistic[0]
        return self._best            

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
        >>> print(perf.statistic())
        {'alg-1': 1.0, 'forest': 0.9500891265597148}     
        """

        data = sorted([(k, self.statistic_func(self.gold, v))
                       for k, v in self.predictions.items()],
                      key=lambda x: x[1], reverse=self.statistic_samples.BiB)
        return dict(data)

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
        >>> print(perf.se())
        {'alg-1': 0.0, 'forest': 0.026945730782184187}
        """
        return SE(self.statistic_samples)

    def plot(self, **kwargs):
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
        if self.score_func is not None:
            value_name = 'Score'
        else:
            value_name = 'Error'
        _ = dict(value_name=value_name)
        _.update(kwargs)  
        return plot_performance(self.statistic_samples, **_)

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
            for key in progress_bar(algs):
                samples(self.gold, self.predictions[key], name=key)
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
    def gold(self):
        """True output, gold standard o :math:`y`"""

        return self._gold

    @gold.setter
    def gold(self, value):
        self._gold = value

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