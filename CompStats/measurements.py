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
import numpy as np
import pandas as pd
from CompStats.bootstrap import StatisticSamples

def CI(samples: np.ndarray, alpha=0.05):
    """Compute the Confidence Interval of a statistic using bootstrap.
    :param samples: Bootstrap samples
    :type samples: np.ndarray
    :param alpha: :math:`[\\frac{\\alpha}{2}, 1 - \\frac{\\alpha}{2}]`. 
    :type alpha: float

    >>> from CompStats import StatisticSamples, CI
    >>> from sklearn.metrics import accuracy_score
    >>> import numpy as np    
    >>> labels = np.r_[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    >>> pred   = np.r_[[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]]
    >>> bootstrap = StatisticSamples(statistic=accuracy_score)
    >>> samples = bootstrap(labels, pred)
    >>> CI(samples)
    (0.6, 1.0)
    """
    alpha = alpha / 2
    return (np.percentile(samples, alpha * 100, axis=0),
            np.percentile(samples, (1 - alpha) * 100, axis=0))

    
def all_differences(statistic_samples: StatisticSamples):
    """Calculates all possible differences in performance among algorithms and sorts by average performance"""
    
    items = list(statistic_samples.calls.items())
    # Calculamos el rendimiento medio y ordenamos los algoritmos basándonos en este
    perf = [(k, v, np.mean(v)) for k, v in items]
    perf.sort(key=lambda x: x[2], reverse=True)  # Orden descendente por rendimiento medio
    
    diffs = {}  # Diccionario para guardar las diferencias
    
    # Iteramos sobre todos los pares posibles de algoritmos ordenados
    for i in range(len(perf)):
        for j in range(i + 1, len(perf)):
            name_i, perf_i, _ = perf[i]
            name_j, perf_j, _ = perf[j]
            
            # Diferencia de i a j
            diff_key_i_to_j = f"{name_i} - {name_j}"
            diffs[diff_key_i_to_j] = np.array(perf_i) - np.array(perf_j)

    
    # Creamos un nuevo objeto StatisticSamples con los resultados
    salida = [(k, np.count_nonzero(v>(2*np.mean(v)))/len(v)) for k, v in diffs.items()]
    #pd.DataFrame.from_dict(salida, orient='index')
    return salida
