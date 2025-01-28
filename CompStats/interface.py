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
from sklearn.metrics import f1_score
from CompStats.bootstrap import StatisticSamples
from CompStats.utils import progress_bar
from CompStats.measurements import SE
from CompStats.performance import plot_performance


def macro(func):
    """Macro score"""

    def inner(y, hy):
        return func(y, hy, average='macro')
    return inner


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
    >>> print(perf)
    Prediction statistics with standard error
    alg-1 = 1.000 (0.000)
    forest = 0.978 (0.019)

    >>> perf_error = clone(perf)
    >>> perf_error.error_func = lambda y, hy: (y != hy).mean()
    >>> print(perf_error)
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
        if len(self.predictions):
            for key, value in progress_bar(self.predictions.items()):
                _(self.gold, value, name=key)
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
        ins.predictions = self.predictions
        ins._statistic_samples.samples = self.statistic_samples.samples
        return ins

    def __str__(self):
        """Prediction statistics with standard error in parenthesis"""

        se = self.se()
        output = ["Prediction statistics with standard error"]
        for key, value in self.statistic().items():
            output.append(f'{key} = {value:0.3f} ({se[key]:0.3f})')
        return "\n".join(output)

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
        return {k: self.statistic_func(self.gold, v)
                for k, v in self.predictions.items()}

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