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
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from typing import Callable
import pandas as pd
import numpy as np
import seaborn as sns
from CompStats.bootstrap import StatisticSamples
from CompStats.measurements import CI


def performance(data: pd.DataFrame,
                gold: str='y',
                score: Callable[[np.ndarray, np.ndarray], float]=accuracy_score,
                statistic_samples: StatisticSamples=None) -> StatisticSamples:
    """Bootstrap samples of a performance score"""
    if statistic_samples is None:
        statistic_samples = StatisticSamples(statistic=score)
    columns = data.columns
    y = data[gold]
    for column in columns:
        if column == gold:
            continue
        statistic_samples(y, data[column], name=column)
    return statistic_samples


def difference(statistic_samples: StatisticSamples, best_index: int=-1):
    """Bootstrap samples of a difference in performnace"""

    items = list(statistic_samples.calls.items())
    perf = [(k, v, np.mean(v)) for k, v in items]
    perf.sort(key=lambda x: x[-1])
    best_name, best_perf, _ = perf[best_index]
    diff = {}
    for alg, alg_perf, _ in perf:
        if alg == best_name:
            continue
        diff[alg] = best_perf - alg_perf
    output = clone(statistic_samples)
    output.calls = diff
    output.info['best'] = best_name
    return output


def all_differences(statistic_samples: StatisticSamples, reverse: bool=True):
    """Calculates all possible differences in performance among algorithms and sorts by average performance"""
    
    items = list(statistic_samples.calls.items())
    # Calculamos el rendimiento medio y ordenamos los algoritmos basándonos en este
    perf = [(k, v, np.mean(v)) for k, v in items]
    perf.sort(key=lambda x: x[2], reverse=reverse)  # Orden descendente por rendimiento medio
    
    diffs = {}  # Diccionario para guardar las diferencias
    
    # Iteramos sobre todos los pares posibles de algoritmos ordenados
    for i in range(len(perf)):
        for j in range(i + 1, len(perf)):
            name_i, perf_i, _ = perf[i]
            name_j, perf_j, _ = perf[j]
            
            # Diferencia de i a j
            diff_key_i_to_j = f"{name_i} - {name_j}"
            diffs[diff_key_i_to_j] = np.array(perf_i) - np.array(perf_j)
    output = clone(statistic_samples)
    output.calls = diffs
    return output
    

def plot_performance(statistic_samples: StatisticSamples,
                     var_name='Algorithm', value_name='Score',
                     capsize=0.2, linestyle='none', kind='point',
                     sharex=False, CI=CI, **kwargs):
    """Plot the performance with the confidence intervals
    
    >>> from CompStats import performance, plot_performance
    >>> from CompStats.tests.test_performance import DATA
    >>> from sklearn.metrics import f1_score
    >>> import pandas as pd
    >>> df = pd.read_csv(DATA)
    >>> score = lambda y, hy: f1_score(y, hy, average='weighted')
    >>> perf = performance(df, score=score)
    >>> ins = plot_performance(perf)
    """

    df2 = pd.DataFrame(statistic_samples.calls).melt(var_name=var_name,
                                                     value_name=value_name)
    f_grid = sns.catplot(df2, x=value_name, y=var_name,
                         capsize=capsize, linestyle=linestyle,
                         kind=kind, errorbar=CI, sharex=sharex, **kwargs)
    return f_grid


def plot_difference(statistic_samples: StatisticSamples,
                    var_name='Comparison', value_name='Difference',
                    set_refline=True, set_title=True,
                    **kwargs):
    """Plot the difference in performance with its confidence intervals
    
    >>> from CompStats import performance, difference, plot_difference
    >>> from CompStats.tests.test_performance import DATA
    >>> from sklearn.metrics import f1_score
    >>> import pandas as pd
    >>> df = pd.read_csv(DATA)
    >>> score = lambda y, hy: f1_score(y, hy, average='weighted')
    >>> perf = performance(df, score=score)
    >>> diff = difference(perf)
    >>> ins = plot_difference(diff)
    """
    f_grid = plot_performance(statistic_samples, var_name=var_name,
                              value_name=value_name, **kwargs)
    if set_refline:
        f_grid.refline(x=0)
    if set_title:
        best = statistic_samples.info['best']
        f_grid.facet_axis(0, 0).set_title(f'Best: {best}')
    return f_grid
