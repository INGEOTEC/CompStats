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
from typing import Callable
from joblib import delayed, Parallel
import numpy as np


class StatisticSamples:
    """Apply the statistic to `num_samples` samples taken with replacement from the population (arguments).

    :param statistic: Statistic.
    :type statistic: Callable
    :param num_samples: Number of bootstrap samples, default=500.
    :type num_samples: int
    :param n_jobs: Number of jobs to run in parallel, default=1.
    :type n_jobs: int


    >>> from IngeoML import StatisticSamples
    >>> from sklearn.metrics import accuracy_score
    >>> import numpy as np
    >>> statistic = StatisticSamples(num_samples=10, statistic=np.mean)
    >>> empirical_distribution = np.r_[[3, 4, 5, 2, 4]]
    >>> statistic(empirical_distribution)
    array([2.8, 3.6, 3.6, 3.6, 2.6, 4. , 2.8, 3. , 3.8, 3.6])
    >>> labels = np.r_[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    >>> pred   = np.r_[[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]]
    >>> acc = StatisticSamples(num_samples=15, statistic=accuracy_score)
    >>> acc(labels, pred)
    array([0.9, 0.8, 0.7, 1. , 0.6, 1. , 0.7, 0.9, 0.9, 0.8, 0.9, 0.8, 0.8, 0.8, 0.8])
    """

    def __init__(self,
                 statistic: Callable[[np.ndarray], float]=np.mean,
                 num_samples: int=500,
                 n_jobs: int=1):
        self.statistic = statistic
        self.num_samples = num_samples
        self.n_jobs = n_jobs
        self._samples = None

    @property
    def n_jobs(self):
        """Number of jobs to do in parallel"""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def statistic(self):
        """Statistic function."""
        return self._statistic
    
    @statistic.setter
    def statistic(self, value):
        self._statistic = value

    @property
    def num_samples(self):
        """Number of bootstrap samples."""
        return self._num_samples
    
    @num_samples.setter
    def num_samples(self, value):
        self._num_samples = value

    @property
    def statistic_samples(self):
        """It contains the statistic samples of the latest call."""
        assert hasattr(self, '_statistic_samples')
        return self._statistic_samples
    
    @statistic_samples.setter
    def statistic_samples(self, value):
        self._statistic_samples = value

    def samples(self, N):
        """Samples.
        
        :param N: Population size.
        :type N: int
        """
        def inner(N):
            _ = np.random.randint(N, size=(self.num_samples, N))
            self._samples = _
            return self._samples
        try:
            if self._samples.shape[1] == N:
                return self._samples
            else:
                return inner(N)
        except AttributeError:
            return inner(N)
  
    def __call__(self, *args: np.ndarray) -> np.ndarray:
        """Population where the bootstrap process will be performed. 

        :param *args: Population
        :type *args: np.ndarray
        """
        def inner(s):
            _ = [arg[s] for arg in args]
            return self.statistic(*_)

        B = []
        # statistic = self.statistic
        B = Parallel(n_jobs=self.n_jobs)(delayed(inner)(s)
                                         for s in self.samples(args[0].shape[0]))
        # for s in self.samples(args[0].shape[0]):
        #     _ = [arg[s] for arg in args]
        #     B.append(statistic(*_))
        self.statistic_samples = np.array(B)
        return self.statistic_samples
   

class CI(StatisticSamples):
    """Compute the Confidence Interval of a statistic using bootstrap.
    
    :param alpha: :math:`[\\frac{\\alpha}{2}, 1 - \\frac{\\alpha}{2}]`. 
    :type alpha: float

    >>> from IngeoML import CI
    >>> from sklearn.metrics import accuracy_score
    >>> import numpy as np    
    >>> labels = np.r_[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    >>> pred   = np.r_[[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]]
    >>> acc = CI(statistic=accuracy_score)
    >>> acc(labels, pred)
    (0.7, 1.0)
    """
    def __init__(self, alpha: float=0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    @property
    def alpha(self):
        """The interval is computed for :math:`[\\frac{\\alpha}{2}, 1 - \\frac{\\alpha}{2}]`.
        """
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = value / 2

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        B =  super().__call__(*args)
        alpha  = self.alpha  
        return (np.percentile(B, alpha * 100, axis=0), 
                np.percentile(B, (1 - alpha) * 100, axis=0))
    

# class SE(StatisticSamples):
#     """Compute the Standard Error of a statistic using bootstrap.

#     >>> from IngeoML import SE
#     >>> from sklearn.metrics import accuracy_score
#     >>> import numpy as np    
#     >>> labels = np.r_[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
#     >>> pred   = np.r_[[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]]
#     >>> se = SE(statistic=accuracy_score)
#     >>> se(labels, pred)
#     0.11949493713124419
#     """

#     def __call__(self, *args: np.ndarray) -> float:
#         B =  super().__call__(*args)
#         return np.std(B, axis=0)


# class Difference(CI):
#     def __init__(self, y: np.ndarray, 
#                  algorithms: dict={}, 
#                  performance: Callable[[np.ndarray, np.ndarray], float]=lambda y, hy: f1_score(y, hy, average='macro'),
#                  **kwargs) -> None:
#         super(Difference, self).__init__(populations=algorithms, statistic=performance)
#         self.y = y
#         self._dist = dict()
#         self._delta = dict()
#         self._pvalue_r = dict()
#         self._pvalue_l = dict()

#     @property
#     def y(self):
#         return self._y
    
#     @y.setter
#     def y(self, value):
#         self._y = value

#     @property
#     def best(self):
#         try:
#             return self._best
#         except AttributeError:
#             y = self.y
#             best = (None, -np.inf)
#             for k, v in self.populations.items():
#                 perf = self.statistic(y, v)
#                 if perf > best[1]:
#                     best = (k, perf)
#             self._best = best[0]
#             return self._best

#     def delta(self, key):
#         assert key != self.best
#         if key in self._delta:
#             return self._delta[key]
#         y = self.y
#         algs = self.populations
#         perf = self.statistic
#         delta = perf(y, algs[self.best]) - perf(y, algs[key])
#         self._delta[key] = delta
#         return delta
    
#     def samples(self, key):
#         if key in self.statistic_samples:
#             return self.statistic_samples[key]
#         data = self.populations[key]
#         y = self.y
#         output = np.array([self.statistic(y[s], data[s])
#                            for s in self.bootstrap])
#         self.statistic_samples[key] = output
#         return output    
    
#     @property
#     def best_performance(self):
#         return self.samples(self.best)
        
#     def distribution(self, key):
#         best = self.best
#         assert key != best
#         if key in self._dist:
#             return self._dist[key]
#         output = self.best_performance - self.samples(key)
#         self._dist[key] = output
#         return output

#     def pvalue(self, key, side='right'):
#         assert side in ['left', 'right']
#         assert key != self.best
#         if side == 'right':
#             if key in self._pvalue_r:
#                 return self._pvalue_r[key]
#         elif key in self._pvalue_l:
#             return self._pvalue_l[key]
#         c = 0
#         delta_2 = 2 * self.delta(key)
#         delta_i = self.distribution(key)
#         if side == 'right':
#             c = (delta_i >= delta_2).mean()
#         else:
#             c = (delta_i < 0).mean()
#         if side == 'right':
#             self._pvalue_r[key] = c
#         else:
#             self._pvalue_l[key] = c
#         return c
    
#     def sort(self, side='right'):
#         best = self.best
#         algs = [(k, self.pvalue(k, side=side))
#                 for k in self.populations if k != best]
#         algs.sort(key=lambda x: x[1], reverse=True)
#         return [k for k, _ in algs]
                