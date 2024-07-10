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
                BiB: bool=True,
                statistic_samples: StatisticSamples=None) -> StatisticSamples:
    """
    Calculate bootstrap samples of a performance score for a given dataset.

    Parameters:
    data (pd.DataFrame): Input dataset.
    gold (str, optional): Column name of the ground truth or target variable. Defaults to 'y'.
    score (Callable, optional): Performance score function. Defaults to accuracy_score.
    num_samples (int, optional): Number of bootstrap samples. Defaults to 500.
    n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
    BiB (bool, optional): Whether the metric is Bigger is Better. Defaults to True.
    statistic_samples (StatisticSamples, optional): Pre-initialized StatisticSamples object. Defaults to None.

    Returns:
    StatisticSamples: Object containing the bootstrap samples of the performance score.

    Example usage:
    >>> from sklearn.metrics import accuracy_score
    >>> import pandas as pd
    >>> from CompStats import performance
    >>> df = pd.read_csv('path/to/data.csv')
    >>> perf = performance(df, gold='y', score=accuracy_score, num_samples=1000)
    """
    if statistic_samples is None:
        statistic_samples = StatisticSamples(statistic=score, num_samples=num_samples,
                                             n_jobs=n_jobs, BiB=BiB)
    columns = data.columns
    y = data[gold]
    for column in progress_bar(columns):
        if column == gold:
            continue
        statistic_samples(y, data[column], name=column)
        
    return statistic_samples


def difference(statistic_samples: StatisticSamples): #, best_index: int=-1):
    """
    Computes the difference in performance between the best performing algorithm and others using bootstrap samples.

    Parameters:
    statistic_samples (StatisticSamples): An instance of StatisticSamples containing the performance data.

    Returns:
    StatisticSamples: A new instance of StatisticSamples with the computed differences and information about the best algorithm.

    The function works as follows:
    1. Determines the index of the best performing algorithm based on the BiB attribute.
    2. Extracts and calculates the mean performance for each algorithm.
    3. Sorts the algorithms by their mean performance.
    4. Identifies the best performing algorithm.
    5. Computes the difference in performance between the best algorithm and each other algorithm.
    6. Returns a new StatisticSamples instance with the computed differences and the name of the best performing algorithm.

    Example usage:
    >>> from CompStats import performance, difference
    >>> from CompStats.tests.test_performance import DATA
    >>> from sklearn.metrics import f1_score
    >>> import pandas as pd
    >>> df = pd.read_csv(DATA)
    >>> score = lambda y, hy: f1_score(y, hy, average='weighted')
    >>> perf = performance(df, score=score)
    >>> diff = difference(perf)
    """
    best_index = -1 if statistic_samples.BiB else 0
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


def all_differences(statistic_samples: StatisticSamples):
    """
    Calculates all possible differences in performance among algorithms and sorts them by average performance.

    Parameters:
    statistic_samples (StatisticSamples): An instance of StatisticSamples containing the performance data.

    Returns:
    StatisticSamples: A new instance of StatisticSamples with the computed performance differences among all algorithms.

    The function works as follows:
    1. Extracts the performance data for each algorithm.
    2. Calculates the mean performance for each algorithm and sorts the algorithms based on their mean performance.
    3. Iterates over all possible pairs of algorithms.
    4. Computes the difference in performance for each pair and stores it in a dictionary.
    5. Returns a new StatisticSamples instance with the computed differences.

    Example usage:
    >>> from CompStats import performance, all_differences
    >>> from CompStats.tests.test_performance import DATA
    >>> from sklearn.metrics import f1_score
    >>> import pandas as pd
    >>> df = pd.read_csv(DATA)
    >>> score = lambda y, hy: f1_score(y, hy, average='weighted')
    >>> perf = performance(df, score=score)
    >>> all_diff = all_differences(perf)
    """
    items = list(statistic_samples.calls.items())
    # Calculamos el rendimiento medio y ordenamos los algoritmos basándonos en este
    perf = [(k, v, np.mean(v)) for k, v in items]
    perf.sort(key=lambda x: x[2], reverse=statistic_samples.BiB)  # Orden por rendimiento medio
    
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
    """
    Plots the performance of algorithms with confidence intervals.

    Parameters:
    statistic_samples (StatisticSamples or pd.DataFrame): An instance of StatisticSamples containing the performance data, 
                                                          or a DataFrame in long format.
    CI (float): Confidence interval level (default is 0.05).
    var_name (str): Variable name for algorithms (default is 'Algorithm').
    value_name (str): Variable name for scores (default is 'Score').
    capsize (float): Size of the caps on error bars (default is 0.2).
    linestyle (str): Line style for the plot (default is 'none').
    kind (str): Type of plot (default is 'point').
    sharex (bool): Whether to share the x-axis among subplots (default is False).
    **kwargs: Additional keyword arguments passed to seaborn's catplot function.

    Returns:
    sns.axisgrid.FacetGrid: A seaborn FacetGrid object containing the plot.

    The function works as follows:
    1. If statistic_samples is an instance of StatisticSamples, it extracts and sorts the performance data.
    2. Converts the data into a long format DataFrame.
    3. Computes the confidence intervals if CI is provided as a float.
    4. Plots the performance data with confidence intervals using seaborn's catplot.

    
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
        lista_ordenada = sorted(statistic_samples.calls.items(), key=lambda x: np.mean(x[1]), reverse=statistic_samples.BiB)
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
    """
    Plot the difference in performance with its confidence intervals.

    Parameters:
    statistic_samples (StatisticSamples): An instance of StatisticSamples containing the performance data.
    CI (float, optional): Confidence interval level. Defaults to 0.05.
    var_name (str, optional): Variable name for the comparisons. Defaults to 'Comparison'.
    value_name (str, optional): Variable name for the differences. Defaults to 'Difference'.
    set_refline (bool, optional): Whether to set a reference line at x=0. Defaults to True.
    set_title (bool, optional): Whether to set the title of the plot with the best performing algorithm. Defaults to True.
    hue (str or None, optional): Column name for hue encoding. Defaults to 'Significant'.
    palette (list or None, optional): Colors to use for different hue levels. Defaults to None.
    **kwargs: Additional keyword arguments passed to the plot_performance function.

    Returns:
    sns.axisgrid.FacetGrid: A seaborn FacetGrid object containing the plot.

    The function works as follows:
    1. Converts the differences stored in statistic_samples into a long format DataFrame.
    2. Adds a 'Significant' column to indicate whether the confidence interval includes zero.
    3. Plots the differences with confidence intervals using the plot_performance function.
    4. Optionally sets a reference line at x=0 and a title indicating the best performing algorithm.
    
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
    for key, (left, right) in measurements.CI(statistic_samples, alpha=CI).items():
        if left < 0 < right:
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
    """
    Calculate bootstrap samples of multiple performance metrics for a given dataset.

    Parameters:
    data (pd.DataFrame): Input dataset.
    gold (str): Column name of the ground truth or target variable.
    scores (List[dict]): A list of dictionaries, each containing:
        - "func": The performance score function.
        - "args" (optional): Arguments to pass to the score function.
        - "BiB": Whether the metric is Bigger is Better.
    num_samples (int, optional): Number of bootstrap samples. Defaults to 500.
    n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.

    Returns:
    dict: A dictionary containing the results for each metric, including:
        - 'samples': Bootstrap samples of the performance scores.
        - 'performance': Calculated performance scores for each algorithm.
        - 'compg': General performance comparison metrics, including:
            - 'n': Number of samples.
            - 'm': Number of algorithms.
            - 'cv': Coefficient of variation for each metric.
            - 'dist': Distance metric for each metric.
            - 'PPI': Performance potential index for each metric.
        - 'BiB': Whether each metric is Bigger is Better.

    The function works as follows:
    1. Defines auxiliary functions for calculating additional performance metrics.
    2. Iterates over the list of score functions and their respective arguments.
    3. Initializes a StatisticSamples object for each score function.
    4. Calculates the performance scores for each column in the dataset (excluding the ground truth column).
    5. Computes additional performance metrics (CV, distance, PPI) for each score function.
    6. Compiles the results into a dictionary and returns it.

    Example usage:
    >>> from sklearn.metrics import accuracy_score, f1_score
    >>> import pandas as pd
    >>> from CompStats import performance_multiple_metrics
    >>> df = pd.read_csv('path/to/data.csv')
    >>> scores = [
    >>>     {"func": accuracy_score, "BiB": True},
    >>>     {"func": f1_score, "args": {"average": "weighted"}, "BiB": True}
    >>> ]
    >>> results = performance_multiple_metrics(df, gold='target', scores=scores, num_samples=1000)
    """
    results, performance_dict, perfo, dist, ccv, cppi, compg, cBiB = {}, {}, {}, {}, {}, {}, {}, {}
    n,m = data.shape
    # definimos las funciones para las metricas
    cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
    dista = lambda x: np.abs(np.max(x) - np.median(x))
    ppi = lambda x: (1 - np.max(x)) * 100
    for score_info in scores:
        score_func = score_info["func"]
        score_args = score_info.get("args", {})
        score_BiB = score_info["BiB"]
        # Prepara el StatisticSamples con los argumentos específicos para esta métrica
        statistic_samples = StatisticSamples(num_samples=num_samples, n_jobs=n_jobs, BiB=score_BiB)
        # Calcula la métrica para cada muestra
        statistic_samples.statistic = statistic = lambda y_true, y_pred: score_func(y_true, y_pred, **score_args)
        # metric_name = score_func.__name__ + "_" + "_".join([f"{key}={value}" for key, value in score_args.items()])
        metric_name = score_func.__name__ + ("" if not score_args else "_" + "_".join([f"{key}={value}" for key, value in score_args.items()]))
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
        cBiB[metric_name] = score_BiB
    compg = {'n' : n,
             'm' : m-1,
             'cv' : ccv,
             'dist' : dist,
             'PPI' : cppi}
    performance_dict = {'samples' : results,
                        'performance' : perfo,
                        'compg' : compg,
                        'BiB': cBiB}
    return performance_dict 

def plot_performance2(results: dict, CI: float=0.05,
                     var_name='Algorithm', value_name='Score',
                     capsize=0.2, linestyle='none', kind='point',
                     sharex=False, **kwargs):
    """
    Plot the performance with confidence intervals. This function is used by plot_performance_difference_multiple

    Parameters:
    results (dict): A dictionary where keys are algorithm names and values are lists of performance scores.
    CI (float, optional): Confidence interval level for error bars. Defaults to 0.05.
    var_name (str, optional): Variable name for the algorithms. Defaults to 'Algorithm'.
    value_name (str, optional): Variable name for the scores. Defaults to 'Score'.
    capsize (float, optional): Cap size for error bars. Defaults to 0.2.
    linestyle (str, optional): Line style for the plot. Defaults to 'none'.
    kind (str, optional): Type of the plot, e.g., 'point', 'bar'. Defaults to 'point'.
    sharex (bool, optional): Whether to share the x-axis among subplots. Defaults to False.
    **kwargs: Additional keyword arguments for seaborn.catplot.

    Returns:
    sns.axisgrid.FacetGrid: A seaborn FacetGrid object containing the plot.

    The function works as follows:
    1. If results is a dictionary, it sorts the algorithms by their mean performance scores.
    2. Converts the sorted data into a long format DataFrame.
    3. Computes the confidence intervals if CI is provided as a float.
    4. Uses seaborn's catplot to create and display the performance plot with confidence intervals.
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

def plot_performance_difference_multiple(results_dict, CI=0.05, capsize=0.2, linestyle='none', kind='point', **kwargs):
    """
    Create multiple performance plots, one for each performance metric in the results dictionary.

    Parameters:
    results_dict (dict): A dictionary where keys are metric names and values are dictionaries 
                         with algorithm names as keys and lists of scores as values.
    CI (float, optional): Confidence interval level for error bars. Defaults to 0.05.
    capsize (float, optional): Cap size for error bars. Defaults to 0.2.
    linestyle (str, optional): Line style for the plot. Defaults to 'none'.
    kind (str, optional): Type of the plot, e.g., 'point', 'bar'. Defaults to 'point'.
    **kwargs: Additional keyword arguments for seaborn.catplot.

    Returns:
    None: The function creates and displays plots.

    The function works as follows:
    1. Iterates over each metric in the results dictionary.
    2. Uses seaborn's catplot to create and display the plot for each metric.
    3. Sets the title of each plot to the metric name and the best performing algorithm.

    Example usage:
    >>> from CompStats import plot_performance_difference_multiple
    >>> import pandas as pd
    >>> from CompStats import performance_multiple_metrics
    >>> df = pd.read_csv('path/to/data.csv')
    >>> scores = [
    >>>     {"func": accuracy_score, "BiB": True},
    >>>     {"func": f1_score, "args": {"average": "weighted"}, "BiB": True}
    >>> ]
    >>> results = performance_multiple_metrics(df, gold='target', scores=scores, num_samples=1000)
    >>> plot_performance_difference_multiple(results, CI=0.05)
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

    Parameters:
    results_dict (dict): A dictionary where keys are metric names and values are dictionaries.
                         Each sub-dictionary has algorithm names as keys and lists of performance scores as values.
    CI (float, optional): Confidence interval level. Defaults to 0.05.

    Returns:
    dict: A dictionary with the same structure, but where the scores for each algorithm are replaced
          by their differences to the scores of the best performing algorithm for that metric,
          excluding the best performing algorithm comparing with itself.
          Also includes the best algorithm name for each metric.

    The function works as follows:
    1. Iterates over each metric in the results dictionary.
    2. Converts performance scores to numpy arrays for efficient computations.
    3. Identifies the best performing algorithm for each metric based on the mean performance scores.
    4. Calculates the differences in performance scores relative to the best performing algorithm.
    5. Computes confidence intervals and p-values for these differences.
    6. Stores the differences, confidence intervals, p-values, and the best algorithm for each metric.
    7. Returns a dictionary with these calculated differences and additional information.

    Example usage:
    >>> from CompStats import performance, difference_multiple
    >>> from CompStats.tests.test_performance import DATA
    >>> from sklearn.metrics import f1_score
    >>> import pandas as pd
    >>> df = pd.read_csv(DATA)
    >>> score = lambda y, hy: f1_score(y, hy, average='weighted')
    >>> perf = performance(df, score=score)
    >>> diff_mult = difference_multiple(perf, CI=0.05)
    """
    differences_dict = results_dict.copy()
    winner = {}
    alpha = CI
    for metric, results in results_dict['samples'].items():
        # Convert scores to arrays for vectorized operations
        scores_arrays = {alg: np.array(scores) for alg, scores in results.items()}
        # Identify the best performing algorithm (highest mean score)
        if results_dict['BiB'][metric]:
            best_alg = max(scores_arrays, key=lambda alg: np.mean(scores_arrays[alg]))
        else:
            best_alg = min(scores_arrays, key=lambda alg: np.mean(scores_arrays[alg]))
        best_scores = scores_arrays[best_alg]
        
        # Calculate differences to the best performing algorithm, excluding the best from comparing with itself
        differences = {alg: best_scores - scores for alg, scores in scores_arrays.items() if alg != best_alg}

        # Calculate Confidence interval for differences to the bet performing algorithm.
        CI_differences = {alg: measurements.CI(np.array(scores), alpha=CI) for alg, scores in differences.items()}
        p_value_differences = {alg: measurements.difference_p_value(np.array(scores), BiB= results_dict['BiB'][metric]) for alg, scores in differences.items()}


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
    for key, (left, right) in diff_dictionary['CI'].items():
        if left < 0 < right:
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

def plot_performance_multiple(results_dict: dict, CI: float = 0.05, capsize: float = 0.2, 
                              linestyle: str = 'none', kind: str = 'point', **kwargs):
    """
    Create multiple performance plots, one for each performance metric in the results dictionary.

    Parameters:
    results_dict (dict): A dictionary where keys are metric names and values are dictionaries 
                         with algorithm names as keys and lists of performance scores as values.
    CI (float, optional): Confidence interval level for error bars. Defaults to 0.05.
    capsize (float, optional): Cap size for error bars. Defaults to 0.2.
    linestyle (str, optional): Line style for the plot. Defaults to 'none'.
    kind (str, optional): Type of the plot, e.g., 'point', 'bar'. Defaults to 'point'.
    **kwargs: Additional keyword arguments for seaborn.catplot.

    Returns:
    None: The function creates and displays plots.

    The function works as follows:
    1. Iterates over each metric in the results dictionary.
    2. Uses the plot_performance2 function to create and display the plot for each metric.
    3. Sets the title of each plot to the metric name and the best performing algorithm.

    Example usage:
    >>> from CompStats import plot_performance_multiple
    >>> results = {
    >>>     'accuracy': {
    >>>         'alg1': [0.1, 0.2, 0.15], 
    >>>         'alg2': [0.05, 0.1, 0.07]
    >>>     },
    >>>     'f1_score': {
    >>>         'alg1': [0.3, 0.25, 0.2], 
    >>>         'alg2': [0.2, 0.15, 0.1]
    >>>     }
    >>> }
    >>> plot_performance_multiple(results, CI=0.05)
    """
    
    for metric_name, metric_results in results_dict['samples'].items():
        # Convert results to long format DataFrame
        df2 = pd.DataFrame(metric_results).melt(var_name='Algorithm', value_name='Score')
        
        # Define the confidence interval function
        if isinstance(CI, float):
            ci = lambda x: measurements.CI(x, alpha=CI)
        
        # Create the plot
        g = sns.catplot(df2, x='Score', y='Algorithm', capsize=capsize, linestyle=linestyle, 
                        kind=kind, errorbar=ci, **kwargs)
        
        # Set the title of the plot
        g.figure.suptitle(metric_name)
        
        # Display the plot
        plt.show()


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
    for metric_name, metric_results in results_dict['winner'].items():
        # Usa catplot para crear y mostrar el gráfico
        g = plot_difference2(metric_results)
        g.figure.suptitle(metric_name)  
        # plt.show()
 



### este por el momento no.
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
        scores_arrays = dict(sorted(scores_arrays.items(), key=lambda item: np.mean(item[1]), reverse=results_dict['BiB'][metric]))

        
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
                p_value_differences[f"{alg_a} vs {alg_b}"] = measurements.difference_p_value(diff, BiB=results_dict['BiB'][metric])
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

