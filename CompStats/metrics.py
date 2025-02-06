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
from sklearn import metrics
from CompStats.interface import Perf
from CompStats.utils import perf_docs


@perf_docs
def accuracy_score(y_true, *y_pred,
                   normalize=True, sample_weight=None,
                   num_samples: int=500,
                   n_jobs: int=-1, 
                   use_tqdm=True,
                   **kwargs):
    """
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.base import clone
    >>> from CompStats.metrics import accuracy_score
    >>> X, y = load_iris(return_X_y=True)
    >>> _ = train_test_split(X, y, test_size=0.3)
    >>> X_train, X_val, y_train, y_val = _
    >>> m = LinearSVC().fit(X_train, y_train)
    >>> hy = m.predict(X_val)
    >>> ens = RandomForestClassifier().fit(X_train, y_train)
    >>> score = accuracy_score(y_val, hy,
                               forest=ens.predict(X_val))
    >>> score
    <Perf>
    Prediction statistics with standard error
    forest = 0.978 (0.023)
    alg-1 = 0.956 (0.030)
    >>> diff = score.difference()
    >>> diff
    <Difference>
    difference p-values w.r.t forest
    alg-1 0.252
    """

    def inner(y, hy):
        return metrics.accuracy_score(y, hy,
                                      normalize=normalize,
                                      sample_weight=sample_weight)
    return Perf(y_true, *y_pred, score_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def balanced_accuracy_score(y_true, *y_pred,
                            sample_weight=None, adjusted=False,
                            num_samples: int=500,
                            n_jobs: int=-1,
                            use_tqdm=True,
                            **kwargs):
    """
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.base import clone
    >>> from CompStats.metrics import balanced_accuracy_score
    >>> X, y = load_iris(return_X_y=True)
    >>> _ = train_test_split(X, y, test_size=0.3)
    >>> X_train, X_val, y_train, y_val = _
    >>> m = LinearSVC().fit(X_train, y_train)
    >>> hy = m.predict(X_val)
    >>> ens = RandomForestClassifier().fit(X_train, y_train)
    >>> score = balanced_accuracy_score(y_val, hy,
                                        forest=ens.predict(X_val))
    >>> score
    <Perf>
    Prediction statistics with standard error
    forest = 0.957 (0.031)
    alg-1 = 0.935 (0.037)
    >>> diff = score.difference()
    >>> diff
    <Difference>
    difference p-values w.r.t forest
    alg-1 0.254  
    """

    def inner(y, hy):
        return metrics.balanced_accuracy_score(y, hy,
                                               adjusted=adjusted,
                                               sample_weight=sample_weight)
    return Perf(y_true, *y_pred, score_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def f1_score(y_true, *y_pred, labels=None, pos_label=1,
             average='binary', sample_weight=None,
             zero_division='warn', num_samples: int=500,
             n_jobs: int=-1, use_tqdm=True,
             **kwargs):
    """
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
    return Perf(y_true, *y_pred, score_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)
