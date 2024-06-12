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
from typing import List, Callable
import pandas as pd
import numpy as np
import seaborn as sns
from CompStats.bootstrap import StatisticSamples
from CompStats.utils import progress_bar
from CompStats import measurements
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


def performance(data: pd.DataFrame,
                gold: str='y',
                score: Callable[[np.ndarray, np.ndarray], float]=accuracy_score,
                num_samples: int=500,
                n_jobs: int=-1,
                statistic_samples: StatisticSamples=None) -> StatisticSamples:
    """Bootstrap samples of a performance score"""
    if statistic_samples is None:
        statistic_samples = StatisticSamples(statistic=score, num_samples=num_samples,
                                             n_jobs=n_jobs)
    columns = data.columns
    y = data[gold]
    for column in progress_bar(columns):
        if column == gold:
            continue
        statistic_samples(y, data[column], name=column)
    mean_values = statistic_samples.mean(axis=0)  
    sorted_columns = mean_values.sort_values(ascending=False).index  
    statistic_samples = statistic_samples.loc[:, sorted_columns]  
    
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
    

def plot_performance(statistic_samples: StatisticSamples, CI: float=0.05,
                     var_name='Algorithm', value_name='Score',
                     capsize=0.2, linestyle='none', kind='point',
                     sharex=False, **kwargs):
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

    if isinstance(statistic_samples, StatisticSamples):
        lista_ordenada = sorted(statistic_samples.calls.items(), key=lambda x: np.mean(x[1]), reverse=True)
        diccionario_ordenado = {nombre: muestras for nombre, muestras in lista_ordenada}
        df2 = pd.DataFrame(diccionario_ordenado).melt(var_name=var_name,
                                                         value_name=value_name)
    else:
        df2 = statistic_samples
    if isinstance(CI, float):
        ci = lambda x: measurements.CI(x, alpha=CI)
    f_grid = sns.catplot(df2, x=value_name, y=var_name,
                         capsize=capsize, linestyle=linestyle,
                         kind=kind, errorbar=ci, sharex=sharex, **kwargs)
    return f_grid


def plot_difference(statistic_samples: StatisticSamples, CI: float=0.05,
                    var_name='Comparison', value_name='Difference',
                    set_refline=True, set_title=True,
                    hue='Significant', palette=None,
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

    df2 = pd.DataFrame(statistic_samples.calls).melt(var_name=var_name,
                                                     value_name=value_name)
    if hue is not None:
        df2[hue] = True
    at_least_one = False
    for key, (left, _) in measurements.CI(statistic_samples, alpha=CI).items():
        if left < 0:
            rows = df2[var_name] == key
            df2.loc[rows, hue] = False
            at_least_one = True
    if at_least_one and palette is None:
        palette = ['r', 'b']
    f_grid = plot_performance(df2, var_name=var_name,
                              value_name=value_name, hue=hue,
                              palette=palette,
                              **kwargs)
    if set_refline:
        f_grid.refline(x=0)
    if set_title:
        best = statistic_samples.info['best']
        f_grid.facet_axis(0, 0).set_title(f'Best: {best}')
    return f_grid

def performance_multiple_metrics(data: pd.DataFrame, gold: str, 
                                 scores: List[dict],
                                 num_samples: int = 500, n_jobs: int = -1):
    results = {}
    performance_dict = {}
    perfo = {}
    dist = {}
    ccv = {}
    cppi = {}
    n,m = data.shape
    compg = {}
    statistic_samples = StatisticSamples(num_samples=num_samples, n_jobs=n_jobs)
    cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
    dista = lambda x: np.abs(np.max(x) - np.median(x))
    ppi = lambda x: (1 - np.max(x)) * 100
    for score_info in scores:
        score_func = score_info["func"]
        score_args = score_info.get("args", {})
        # Prepara el StatisticSamples con los argumentos específicos para esta métrica
        statistic_samples.statistic = statistic = lambda y_true, y_pred: score_func(y_true, y_pred, **score_args)
        metric_name = score_func.__name__ + "_" + "_".join([f"{key}={value}" for key, value in score_args.items()])
        results[metric_name] = {}
        perfo[metric_name] = {}
        for column in data.columns:
            if column == gold:
                continue
            results[metric_name][column] = statistic_samples(data[gold], data[column])
            perfo[metric_name][column]  = statistic(data[gold], data[column])
        ccv[metric_name] = cv(np.array(list(perfo[metric_name].values())))
        dist[metric_name] = dista(np.array(list(perfo[metric_name].values())))
        cppi[metric_name] = ppi(np.array(list(perfo[metric_name].values())))
    compg = {'n' : n,
             'm' : m-1,
             'cv' : ccv,
             'dist' : dist,
             'PPI' : cppi}
    performance_dict = {'samples' : results,
                        'performance' : perfo,
                        'compg' : compg}
    return performance_dict 

def plot_performance2(results: dict, CI: float=0.05,
                     var_name='Algorithm', value_name='Score',
                     capsize=0.2, linestyle='none', kind='point',
                     sharex=False, **kwargs):
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

    if isinstance(results, dict):
        lista_ordenada = sorted(results.items(), key=lambda x: np.mean(x[1]), reverse=True)
        diccionario_ordenado = {nombre: muestras for nombre, muestras in lista_ordenada}
        df2 = pd.DataFrame(diccionario_ordenado).melt(var_name=var_name,
                                                         value_name=value_name)

    if isinstance(CI, float):
        ci = lambda x: measurements.CI(x, alpha=CI)
    f_grid = sns.catplot(df2, x=value_name, y=var_name,
                         capsize=capsize, linestyle=linestyle,
                         kind=kind, errorbar=ci, sharex=sharex, **kwargs)
    return f_grid

def plot_performance_multiple(results_dict, CI=0.05, capsize=0.2, linestyle='none', kind='point', **kwargs):
    """
    Create multiple performance plots, one for each performance metric in the results dictionary.
    
    :param results_dict: A dictionary where keys are metric names and values are dictionaries with algorithm names as keys and lists of scores as values.
    :param CI: Confidence interval level for error bars.
    :param capsize: Cap size for error bars.
    :param linestyle: Line style for the plot.
    :param kind: Type of the plot, e.g., 'point', 'bar'.
    :param kwargs: Additional keyword arguments for seaborn.catplot.
    """   
    for metric_name, metric_results in results_dict['winner'].items():
        # Usa catplot para crear y mostrar el gráfico
        g = plot_performance2(metric_results['diff'], CI=CI)
        g.figure.suptitle(metric_name+'('+metric_results['best']+')')  
        # plt.show()
 

def difference_multiple(results_dict, CI: float=0.05,):
    """
    Calculate performance differences for multiple metrics, excluding the comparison of the best
    with itself. Additionally, identify the best performing algorithm for each metric.
    
    :param results_dict: A dictionary where keys are metric names and values are dictionaries.
                         Each sub-dictionary has algorithm names as keys and lists of performance scores as values.
    :return: A dictionary with the same structure, but where the scores for each algorithm are replaced
             by their differences to the scores of the best performing algorithm for that metric,
             excluding the best performing algorithm comparing with itself.
             Also includes the best algorithm name for each metric.
    """
    differences_dict = results_dict.copy()
    winner = {}
    alpha = CI
    for metric, results in results_dict['samples'].items():
        # Convert scores to arrays for vectorized operations
        scores_arrays = {alg: np.array(scores) for alg, scores in results.items()}
        # Identify the best performing algorithm (highest mean score)
        best_alg = max(scores_arrays, key=lambda alg: np.mean(scores_arrays[alg]))
        best_scores = scores_arrays[best_alg]
        
        # Calculate differences to the best performing algorithm, excluding the best from comparing with itself
        differences = {alg: best_scores - scores for alg, scores in scores_arrays.items() if alg != best_alg}

        # Calculate Confidence interval for differences to the bet performing algorithm.
        CI_differences = {alg: measurements.CI(np.array(scores), alpha=CI) for alg, scores in differences.items()}
        p_value_differences = {alg: measurements.difference_p_value(np.array(scores)) for alg, scores in differences.items()}


        # Store the differences and the best algorithm under the current metric
        winner[metric] = {'best': best_alg, 'diff': differences,'CI':CI_differences,
                                    'p_value': p_value_differences,
                                    'none': sum(valor > alpha for valor in p_value_differences.values()),
                                    'bonferroni': sum(multipletests(list(p_value_differences.values()), method='bonferroni')[1] > alpha), 
                                    'holm': sum(multipletests(list(p_value_differences.values()), method='holm')[1] > alpha),
                                    'HB': sum(multipletests(list(p_value_differences.values()), method='fdr_bh')[1] > alpha) }
    differences_dict['winner'] = winner
    return differences_dict


def plot_difference2(diff_dictionary: dict, 
                    var_name='Comparison', value_name='Difference',
                    set_refline=True, set_title=True,
                    hue='Significant', palette=None,
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

    df2 = pd.DataFrame(diff_dictionary['diff']).melt(var_name=var_name,
                                                     value_name=value_name)
    if hue is not None:
        df2[hue] = True
    at_least_one = False
    for key, (left, _) in diff_dictionary['CI'].items():
        if left < 0:
            rows = df2[var_name] == key
            df2.loc[rows, hue] = False
            at_least_one = True
    if at_least_one and palette is None:
        palette = ['r', 'b']
    f_grid = plot_performance(df2, var_name=var_name,
                              value_name=value_name, hue=hue,
                              palette=palette,
                              **kwargs)
    if set_refline:
        f_grid.refline(x=0)
    if set_title:
        best = diff_dictionary['best']
        f_grid.facet_axis(0, 0).set_title(f'Best: {best}')
    return f_grid



def plot_difference_multiple(results_dict, CI=0.05, capsize=0.2, linestyle='none', kind='point', **kwargs):
    """
    Create multiple performance plots, one for each performance metric in the results dictionary.
    
    :param results_dict: A dictionary where keys are metric names and values are dictionaries with algorithm names as keys and lists of scores as values.
    :param CI: Confidence interval level for error bars.
    :param capsize: Cap size for error bars.
    :param linestyle: Line style for the plot.
    :param kind: Type of the plot, e.g., 'point', 'bar'.
    :param kwargs: Additional keyword arguments for seaborn.catplot.
    """   
    for metric_name, metric_results in results_dict.items():
        # Usa catplot para crear y mostrar el gráfico
        g = plot_difference2(metric_results)
        g.figure.suptitle(metric_name)  
        # plt.show()
 


def plot_difference_scatter_multiple(results_dict,algorithm: str):
    dict = {}
    for metric_name, metric_results in results_dict.items():
        # Usa catplot para crear y mostrar el gráfico
        g = plot_difference2(metric_results)
        g.figure.suptitle(metric_name)  
        # plt.show()



def plot_scatter_matrix(perf):
    """
    Generate a scatter plot matrix comparing the performance of the same algorithm
    across different metrics contained in the 'perf' dictionary.
    
    :param perf: A dictionary where keys are metric names and values are dictionaries with algorithm names as keys
                 and lists of performance scores as values.
    """
    # Convertir 'perf' en un DataFrame de pandas para facilitar la manipulación
    df_long = pd.DataFrame([
        {"Metric": metric, "Algorithm": alg, "Score": score, "Indice": i}
        for metric, alg_scores in perf.items()
        for alg, scores in alg_scores.items()
        for i, (score)  in enumerate(scores)
        ])
    df_wide = df_long.pivot(index=['Algorithm','Indice'],columns='Metric',values='Score')
    df_wide = df_wide.reset_index(level=[0])
    sns.pairplot(df_wide, diag_kind='kde',hue="Algorithm", corner=True)
    plt.suptitle('Scatter Plot Matrix of Algorithms Performance Across Different Metrics', y=1.02)
    plt.show()



def unique_pairs_differences(results_dict, alpha: float=0.05):
    """
    Calculate performance differences for unique pairs of algorithms for multiple metrics.
    Also, calculates the confidence interval for the differences.
    
    :param results_dict: A dictionary where keys are metric names and values are dictionaries.
                         Each sub-dictionary has algorithm names as keys and lists of performance scores as values.
    :return: A dictionary where each metric name maps to another dictionary.
             This dictionary contains keys for unique pairs of algorithms and their performance differences,
             including the confidence interval for these differences.
    """
    differences_dict = results_dict.copy()
    all = {}
    for metric, results in results_dict['samples'].items():
        # Convert scores to arrays for vectorized operations
        scores_arrays = {alg: np.array(scores) for alg, scores in results.items()}
        
        differences = {}
        p_value_differences = {}
        
        algorithms = list(scores_arrays.keys())
        # Calculate differences for unique pairs of algorithms
        for i, alg_a in enumerate(algorithms):
            for alg_b in algorithms[i+1:]:  # Start from the next algorithm to avoid duplicate comparisons
                # Calculate the difference between alg_a and alg_b
                diff = scores_arrays[alg_a] - scores_arrays[alg_b]
                differences[f"{alg_a} vs {alg_b}"] = diff
                
                # Placeholder for confidence interval calculation
                # Replace the string with an actual call to your CI calculation function
                p_value_differences[f"{alg_a} vs {alg_b}"] = measurements.difference_p_value(diff)
                # For example:
                # CI_differences[f"{alg_a} vs {alg_b}"] = measurements.CI(diff, alpha=CI)
                
        # Store the differences under the current metric
        all[metric] = {'diff': differences, 'p_value': p_value_differences, 
                                    'none': sum(valor > alpha for valor in p_value_differences.values()),
                                    'bonferroni': sum(multipletests(list(p_value_differences.values()), method='bonferroni')[1] > alpha), 
                                    'holm': sum(multipletests(list(p_value_differences.values()), method='holm')[1] > alpha),
                                    'HB': sum(multipletests(list(p_value_differences.values()), method='fdr_bh')[1] > alpha)  }
    differences_dict['all'] = all
    return differences_dict

