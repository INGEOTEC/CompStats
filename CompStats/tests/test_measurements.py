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
import os
from CompStats.measurements import CI, all_differences
from CompStats.bootstrap import StatisticSamples
from CompStats.performance import performance, plot_performance, difference, plot_difference
from sklearn.metrics import f1_score

DATA = os.path.join(os.path.dirname(__file__), 'data.csv')

def test_CI():
    """Test confidence interval"""

    statistic = StatisticSamples(num_samples=26, n_jobs=-1)
    pop = np.r_[3, 4, 5, 2, 4]
    samples = statistic(pop)
    low, high = CI(samples)
    mean = pop.mean()
    assert low < mean < high


def test_all_differences():
    """Test all_differences"""
    df = pd.read_csv(DATA)
    perf = performance(df, score=lambda y, hy: f1_score(y, hy, average='weighted'))
    res = all_differences(perf)
    assert 0 < res[0][1] < 1
