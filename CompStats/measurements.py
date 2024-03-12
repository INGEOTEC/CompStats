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
    if isinstance(samples, StatisticSamples):
        return {k: CI(v, alpha=alpha) for k, v in samples.calls.items()}
    alpha = alpha / 2
    return (np.percentile(samples, alpha * 100, axis=0),
            np.percentile(samples, (1 - alpha) * 100, axis=0))


def SE(samples: np.ndarray):
    """Compute the Standard Error of a statistic using bootstrap.
    
    >>> from CompStats import StatisticSamples, SE
    >>> from sklearn.metrics import accuracy_score
    >>> import numpy as np    
    >>> labels = np.r_[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    >>> pred   = np.r_[[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]]
    >>> bootstrap = StatisticSamples(statistic=accuracy_score)
    >>> samples = bootstrap(labels, pred)
    >>> SE(samples)
    """
    if isinstance(samples, StatisticSamples):
        return {k: SE(v) for k, v in samples.calls.items()}
    return np.std(samples, axis=0)

    
def difference_p_value(statistic_samples: StatisticSamples):
    """Compute the difference p-value"""
    return {k: (v > 2 * np.mean(v)).mean()
            for k, v in statistic_samples.calls.items()}