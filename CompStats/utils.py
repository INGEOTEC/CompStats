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


def progress_bar(arg, use_tqdm: bool=True, **kwargs):
    """Progress bar using tqdm"""
    if not USE_TQDM or not use_tqdm:
        return arg
    return tqdm(arg, **kwargs)


def metrics_docs(hy_name='y_pred', attr_name='score_func'):
    """Decorator to set docs"""

    def perf_docs(func):
        """Decorator to Perf to write :py:class:`~sklearn.metrics` documentation"""

        func.__doc__ = f""":py:class:`~CompStats.interface.Perf` with :py:func:`~sklearn.metrics.{func.__name__}` as :py:attr:`{attr_name}.` The parameters not described can be found in :py:func:`~sklearn.metrics.{func.__name__}`.
        
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

    """ + func.__doc__

        @wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner
    return perf_docs


def dataframe(instance, value_name:str='Score',
              var_name:str='Performance',
              alg_legend:str='Algorithm',
              perf_names:list=None):
    """Dataframe"""
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