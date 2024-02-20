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
    if statistic_samples is None:
        statistic_samples = StatisticSamples(statistic=score)
    columns = data.columns
    y = data[gold]
    for column in columns:
        if column == gold:
            continue
        statistic_samples(y, data[column], name=column)
    return statistic_samples


def plot_performance(statistic_samples: StatisticSamples,
                     var_name='Algorithm', value_name='Score',
                     capsize=0.2, linestyle='none', kind='point',
                     sharex=False, **kwargs):
    """Plot the performance with the confidence intervals"""

    df2 = pd.DataFrame(statistic_samples.calls).melt(var_name=var_name,
                                                     value_name=value_name)

    f_grid = sns.catplot(df2, x=value_name, y=var_name,
                         capsize=capsize, linestyle=linestyle, 
                         kind=kind, errorbar=CI, sharex=sharex, **kwargs)
    return f_grid