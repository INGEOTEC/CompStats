# Copyright 2025 Sergio Nava Muñoz and Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from CompStats.interface import Perf
from sklearn import metrics


def f1_score(y_true, *args, labels=None, pos_label=1,
             average='binary', sample_weight=None,
             zero_division='warn', num_samples: int=500,
             n_jobs: int=-1, **kwargs):
    """:py:class:`~CompStats.interface.Perf` with :py:func:`~sklearn.metrics.f1_score` as :py:attr:`score_func.`

    :param y_true: True measurement or could be a pandas.DataFrame where column label 'y' corresponds to the true measurement.
    :type y_true: numpy.ndarray or pandas.DataFrame 
    :param args: Predictions, the algorithms will be identified with alg-k where k=1 is the first argument included in :py:attr:`args.`
    :type args: numpy.ndarray
    :param kwargs: Predictions, the algorithms will be identified using the keyword
    :type args: numpy.ndarray
    :param n_jobs: Number of jobs to compute the statistic, default=-1 corresponding to use all threads.
    :type n_jobs: int
    :param num_samples: Number of bootstrap samples, default=500.
    :type num_samples: int
    :param labels: see sklearn.metrics.f1_score
    :param pos_label: see sklearn.metrics.f1_score
    :param average: see sklearn.metrics.f1_score
    :param sample_weight: see sklearn.metrics.f1_score
    :param zero_division: see sklearn.metrics.f1_score

    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.base import clone
    >>> from CompStats.metrics import f1_score
    >>> X, y = load_iris(return_X_y=True)
    >>> _ = train_test_split(X, y, test_size=0.3)
    >>> X_train, X_val, y_train, y_val = _
    >>> m = LinearSVC().fit(X_train, y_train)
    >>> hy = m.predict(X_val)
    >>> ens = RandomForestClassifier().fit(X_train, y_train)
    >>> score = f1_score(y_val, hy,
                         forest=ens.predict(X_val),
                         average='macro')
    >>> score
    <Perf>
    Prediction statistics with standard error
    forest = 0.954 (0.032)
    alg-1 = 0.931 (0.040)
    >>> diff = score.difference()
    >>> diff
    <Difference>
    difference p-values w.r.t forest
    alg-1 0.176   
    """

    def inner(y, hy):
        return metrics.f1_score(y, hy, labels=labels,
                                pos_label=pos_label,
                                average=average,
                                sample_weight=sample_weight,
                                zero_division=zero_division)
    return Perf(y_true, *args, score_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                **kwargs)