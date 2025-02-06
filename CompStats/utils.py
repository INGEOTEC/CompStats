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
try:
    USE_TQDM = True
    from tqdm import tqdm
except ImportError:
    USE_TQDM = False


def progress_bar(arg, use_tqdm: bool=True, **kwargs):
    """Progress bar using tqdm"""
    if not USE_TQDM or not use_tqdm:
        return arg
    return tqdm(arg, **kwargs)


from functools import wraps


def perf_docs(func):
    """Decorator to Perf with any write :py:class:`~sklearn.metrics` documentation
    """

    func.__doc__ = f""":py:class:`~CompStats.interface.Perf` with :py:func:`~sklearn.metrics.{func.__name__}` as :py:attr:`score_func.` The parameters not described can be found in :py:func:`~sklearn.metrics.{func.__name__}`.
    
:param y_true: True measurement or could be a pandas.DataFrame where column label 'y' corresponds to the true measurement.
:type y_true: numpy.ndarray or pandas.DataFrame 
:param y_pred: Predictions, the algorithms will be identified with alg-k where k=1 is the first argument included in :py:attr:`y_pred.`
:type y_pred: numpy.ndarray
:param kwargs: Predictions, the algorithms will be identified using the keyword
:type kwargs: numpy.ndarray
:param num_samples: Number of bootstrap samples, default=500.
:type num_samples: int
:param n_jobs: Number of jobs to compute the statistic, default=-1 corresponding to use all threads.
:type n_jobs: int
:param use_tqdm: Whether to use tqdm.tqdm to visualize the progress, default=True
:type use_tqdm: bool
""" + func.__doc__

    @wraps(func)
    def inner(*args, **kwargs):
        return func(*args, **kwargs)

    return inner