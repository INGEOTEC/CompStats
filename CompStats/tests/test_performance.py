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
from sklearn.metrics import f1_score
import seaborn as sns
from CompStats.performance import performance, plot_performance, difference, plot_difference, all_differences


DATA = os.path.join(os.path.dirname(__file__), 'data.csv')


def test_performance():
    df = pd.read_csv(DATA)
    perf = performance(df, score=lambda y, hy: f1_score(y, hy, average='weighted'))
    assert 'BoW' in perf.calls
    assert 'y' not in perf.calls
    assert perf.n_jobs == -1
    

def test_plot_performance():
    df = pd.read_csv(DATA)
    perf = performance(df, score=lambda y, hy: f1_score(y, hy, average='weighted'))
    ins = plot_performance(perf)
    assert isinstance(ins, sns.FacetGrid)


def test_difference():
    """Test difference"""
    df = pd.read_csv(DATA)
    perf = performance(df, score=lambda y, hy: f1_score(y, hy, average='weighted'))
    res = difference(perf)
    assert 'INGEOTEC' not in res.calls
    assert res.info['best'] == 'INGEOTEC'
    ins = plot_difference(res)
    assert isinstance(ins, sns.FacetGrid)


def test_all_differences():
    """Test all_differences"""
    df = pd.read_csv(DATA)
    perf = performance(df, score=lambda y, hy: f1_score(y, hy, average='weighted'))
    res = all_differences(perf)
    assert 'INGEOTEC - BoW' in res.calls