# Copyright 2023 Mario Graff Guerrero

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
from CompStats.bootstrap import StatisticSamples


def problem_algorithms():
    labels = [0, 0, 0, 0, 0,
                1, 1, 1, 1, 1]
    a = [0, 0, 0, 0, 0,
            1, 1, 1, 1, 0]
    b = [0, 0, 1, 0, 0,
            1, 1, 1, 1, 0]
    c = [0, 0, 0, 1, 0,
            1, 1, 0, 1, 0]
    return (np.array(labels),
            dict(a=np.array(a),
                    b=np.array(b),
                    c=np.array(c)))


def test_StatisticSample():
    """Test StatisticSamples"""

    statistic = StatisticSamples(num_samples=26, n_jobs=-1)
    indexes = statistic.samples(5)
    samples = statistic(np.r_[3, 4, 5, 2, 4])
    assert samples.shape[0] == 26
    assert np.fabs(indexes - statistic.samples(5)).sum() == 0


def test_StatisticSample_name():
    """Test the storing feature"""

    statistic = StatisticSamples(num_samples=26, n_jobs=-1)
    indexes = statistic.samples(5)
    samples = statistic(np.r_[3, 4, 5, 2, 4], name='first')
    assert np.fabs(samples - statistic['first']).sum() == 0



# def test_CI():
#     """Test CI"""
#     statistic = CI()
#     ci = statistic(np.r_[[3, 4, 5, 2, 4]])
#     assert len(ci) == 2


# def test_CI2D():
#     """Test CI with two values"""
#     from sklearn.metrics import f1_score
#     labels = np.r_[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0]]
#     pred   = np.r_[[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]]
#     ci = CI(statistic=lambda y, hy: f1_score(y, hy, average=None))
#     a = ci(labels, pred)
#     assert a[0].shape[0] == 2 and a[1].shape[0] == 2


# def test_se():
#     labels = np.r_[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
#     pred   = np.r_[[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]]
#     se = SE(statistic=accuracy_score)
#     res = se(labels, pred)
#     assert res > 0 and isinstance(res, float)

# def test_Difference_ci():
#     labels, algs = problem_algorithms()
#     diff = Difference(labels, algs)
#     a = diff.confidence_interval('a')
#     assert a[0] > 0.6 and a[1] <= 1.0


# def test_Difference_best():
#     labels, algs = problem_algorithms()
#     diff = Difference(labels, algs)
#     assert diff.best == 'a'


# def test_Difference_delta():
#     labels, algs = problem_algorithms()
#     diff = Difference(labels, algs)
#     assert diff.delta('b') > 0 and diff.delta('c') > 0


# def test_Difference():
#     labels, algs = problem_algorithms()
#     diff = Difference(labels, algs)
#     assert diff.best == 'a'
#     assert diff.pvalue('b') > diff.pvalue('c')


# def test_Difference_sort():
#     labels, algs = problem_algorithms()
#     diff = Difference(labels, algs)
#     for x, r in zip(diff.sort(), ['b', 'c']):
#         assert x == r