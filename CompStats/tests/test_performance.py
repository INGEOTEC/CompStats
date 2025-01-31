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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import seaborn as sns
from CompStats import performance, plot_performance, difference, plot_difference, all_differences
from CompStats import performance_multiple_metrics, plot_performance2
from CompStats import difference_multiple, plot_scatter_matrix, all_differences_multiple, plot_performance_multiple
from CompStats import plot_difference2, plot_difference_multiple



DATA = os.path.join(os.path.dirname(__file__), 'data.csv')


def test_performance():
    """Test performance"""
    df = pd.read_csv(DATA)
    perf = performance(df, score=lambda y, hy: f1_score(y, hy, average='weighted'))
    assert 'BoW' in perf.calls
    assert 'y' not in perf.calls
    assert perf.n_jobs == -1
    

def test_plot_performance():
    """Test plot_performance"""
    df = pd.read_csv(DATA)
    perf = performance(df, n_jobs=1,
                       score=lambda y, hy: f1_score(y, hy, average='weighted'))
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
    resa = all_differences(perf)
    assert 'INGEOTEC - BoW' in resa.calls


def test_performance_multiple_metrics():
    """Test performance_multiple_metrics"""
    df = pd.read_csv(DATA)
    metrics = [
        {"func": accuracy_score, 'BiB': True},
        {"func": f1_score, "args": {"average": "macro"}, 'BiB': True},
        {"func": precision_score, "args": {"average": "macro"}, 'BiB': True},
        {"func": mean_absolute_error, 'BiB': False}
        ]
    perf = performance_multiple_metrics(df, "y", metrics)
    plot_performance_multiple(perf)
    assert 'accuracy_score' in perf['samples']
    assert 'y' not in perf['samples']['accuracy_score']
    assert 'INGEOTEC' in perf['samples']['accuracy_score']


def test_difference_multiple():
    """Test difference_multiple"""
    df = pd.read_csv(DATA)
    metrics = [
        {"func": accuracy_score, 'BiB': True},
        {"func": f1_score, "args": {"average": "macro"}, 'BiB': True},
        {"func": precision_score, "args": {"average": "macro"}, 'BiB': True},
        {"func": mean_absolute_error, 'BiB': False}
        ]
    perf = performance_multiple_metrics(df, "y", metrics)
    diff = difference_multiple(perf)
    plot_difference_multiple(diff)
    assert diff['winner']['accuracy_score']['best'] == 'BoW'
    assert 'BoW' not in diff['winner']['accuracy_score']['diff'].keys()
    # assert isinstance(ins, sns.FacetGrid)


def test_difference_summary():
    """Test difference_summary"""
    df = pd.read_csv(DATA)
    metrics = [
        {"func": accuracy_score, 'BiB': True},
        {"func": f1_score, "args": {"average": "macro"}, 'BiB': True},
        {"func": precision_score, "args": {"average": "macro"}, 'BiB': True},
        {"func": mean_absolute_error, 'BiB': False}
        ]
    perf = performance_multiple_metrics(df, "y", metrics)
    diff = difference_multiple(perf)
    all_dif = all_differences_multiple(diff)
    assert diff['winner']['accuracy_score']['best'] == 'BoW'
    assert 'BoW' not in diff['winner']['accuracy_score']['diff'].keys()
    assert all_dif['all']['accuracy_score']['none'] == 6
