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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import metrics


def test_f1_score():
    """Test f1_score"""
    from CompStats.metrics import f1_score

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = f1_score(y_val, forest=hy,
                    num_samples=50, average='macro')
    assert 'forest' in perf.statistic
    _ = metrics.f1_score(y_val, hy, average='macro')
    assert _ == perf.statistic['forest']
    perf = f1_score(y_val, hy, average=None)
    assert str(perf) is not None
    nb = GaussianNB().fit(X_train, y_train)
    perf(nb.predict(X_val))
    assert str(perf) is not None


def test_accuracy_score():
    """Test f1_score"""
    from CompStats.metrics import accuracy_score

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = accuracy_score(y_val, forest=hy,
                          num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.accuracy_score(y_val, hy)
    assert _ == perf.statistic['forest']


def test_balanced_accuracy_score():
    """Test f1_score"""
    from CompStats.metrics import balanced_accuracy_score

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = balanced_accuracy_score(y_val, forest=hy,
                                   num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.balanced_accuracy_score(y_val, hy)
    assert _ == perf.statistic['forest']


def test_top_k_accuracy_score():
    """Test top_k_accuracy_score"""
    from CompStats.metrics import top_k_accuracy_score

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, stratify=y)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict_proba(X_val)
    perf = top_k_accuracy_score(y_val,
                                forest=hy,
                                num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.top_k_accuracy_score(y_val, hy)
    assert _ == perf.statistic['forest']


def test_average_precision_score():
    """Test average_precision_score"""
    from CompStats.metrics import average_precision_score

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, stratify=y)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict_proba(X_val)
    perf = average_precision_score(y_val,
                                forest=hy,
                                num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.average_precision_score(y_val, hy)
    assert _ == perf.statistic['forest']


def test_brier_score_loss():
    """Test brier_score_loss"""
    from CompStats.metrics import brier_score_loss
    import numpy as np

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, stratify=y)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict_proba(X_val)[:, 0]
    perf = brier_score_loss(np.where(y_val == 0, 1, 0),
                            forest=hy,
                            num_samples=50)
    nb = GaussianNB().fit(X_train, y_train)
    perf(nb.predict_proba(X_val)[:, 0])
    assert 'forest' in perf.statistic
    _ = metrics.brier_score_loss(np.where(y_val == 0, 1, 0), hy)
    assert _ == perf.statistic['forest']


def test_log_loss():
    """Test log_loss"""
    from CompStats.metrics import log_loss
    import numpy as np

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, stratify=y)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict_proba(X_val)
    perf = log_loss(y_val,
                    forest=hy,
                    num_samples=50)
    nb = GaussianNB().fit(X_train, y_train)
    perf(nb.predict_proba(X_val))
    assert 'forest' in perf.statistic
    _ = metrics.log_loss(y_val, hy)
    assert _ == perf.statistic['forest']


def test_precision_score():
    """Test precision_score"""
    from CompStats.metrics import precision_score
    import numpy as np

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, stratify=y)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = precision_score(y_val,
                           forest=hy,
                           num_samples=50, average='macro')
    assert 'forest' in perf.statistic
    _ = metrics.precision_score(y_val, hy, average='macro')
    assert _ == perf.statistic['forest']


def test_recall_score():
    """Test recall_score"""
    from CompStats.metrics import recall_score
    import numpy as np

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, stratify=y)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = recall_score(y_val,
                           forest=hy,
                           num_samples=50, average='macro')
    assert 'forest' in perf.statistic
    _ = metrics.recall_score(y_val, hy, average='macro')
    assert _ == perf.statistic['forest']


def test_jaccard_score():
    """jaccard_score"""
    from CompStats.metrics import jaccard_score
    import numpy as np

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, stratify=y)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = jaccard_score(y_val,
                         forest=hy,
                         num_samples=50, average='macro')
    assert 'forest' in perf.statistic
    _ = metrics.jaccard_score(y_val, hy, average='macro')
    assert _ == perf.statistic['forest']


def test_roc_auc_score():
    """roc_auc_score"""
    from CompStats.metrics import roc_auc_score
    import numpy as np

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, stratify=y)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict_proba(X_val)
    perf = roc_auc_score(y_val,
                         forest=hy, multi_class='ovr',
                         num_samples=50, average='macro')
    assert 'forest' in perf.statistic
    _ = metrics.roc_auc_score(y_val, hy, multi_class='ovr',
                              average='macro')
    assert _ == perf.statistic['forest']


def test_d2_log_loss_score():
    """d2_log_loss_score"""
    from CompStats.metrics import d2_log_loss_score
    import numpy as np

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, stratify=y)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy = ens.predict_proba(X_val)
    perf = d2_log_loss_score(y_val,
                         forest=hy,
                         num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.d2_log_loss_score(y_val, hy)
    assert _ == perf.statistic['forest']


def test_explained_variance_score():
    """explained_variance_score"""
    from CompStats.metrics import explained_variance_score

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = explained_variance_score(y_val,
                                    forest=hy,
                                    num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.explained_variance_score(y_val, hy)
    assert _ == perf.statistic['forest']


def test_max_error():
    """max_error"""
    from CompStats.metrics import max_error

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = max_error(y_val,
                     forest=hy,
                     num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.max_error(y_val, hy)
    assert _ == perf.statistic['forest']


def test_mean_absolute_error():
    """mean_absolute_error"""
    from CompStats.metrics import mean_absolute_error

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = mean_absolute_error(y_val,
                               forest=hy,
                               num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.mean_absolute_error(y_val, hy)
    assert _ == perf.statistic['forest']


def test_mean_squared_error():
    """mean_squared_error"""
    from CompStats.metrics import mean_squared_error

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = mean_squared_error(y_val,
                               forest=hy,
                               num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.mean_squared_error(y_val, hy)
    assert _ == perf.statistic['forest']


def test_root_mean_squared_error():
    """root_mean_absolute_error"""
    from CompStats.metrics import root_mean_squared_error

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = root_mean_squared_error(y_val,
                                   forest=hy,
                                   num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.root_mean_squared_error(y_val, hy)
    assert _ == perf.statistic['forest']


def test_mean_squared_log_error():
    """mean_squared_log_error"""
    from CompStats.metrics import mean_squared_log_error

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = mean_squared_log_error(y_val,
                                   forest=hy,
                                   num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.mean_squared_log_error(y_val, hy)
    assert _ == perf.statistic['forest']


def test_root_mean_squared_log_error():
    """root_mean_squared_log_error"""
    from CompStats.metrics import root_mean_squared_log_error

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = root_mean_squared_log_error(y_val,
                                       forest=hy,
                                       num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.root_mean_squared_log_error(y_val, hy)
    assert _ == perf.statistic['forest']


def test_median_absolute_error():
    """median_absolute_error"""
    from CompStats.metrics import median_absolute_error

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = median_absolute_error(y_val,
                                       forest=hy,
                                       num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.median_absolute_error(y_val, hy)
    assert _ == perf.statistic['forest']


def test_r2_score():
    """r2_score"""
    from CompStats.metrics import r2_score

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = r2_score(y_val,
                                       forest=hy,
                                       num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.r2_score(y_val, hy)
    assert _ == perf.statistic['forest']


def test_mean_poisson_deviance():
    """mean_poisson_deviance"""
    from CompStats.metrics import mean_poisson_deviance

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = mean_poisson_deviance(y_val,
                                 forest=hy,
                                 num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.mean_poisson_deviance(y_val, hy)
    assert _ == perf.statistic['forest']


def test_mean_gamma_deviance():
    """mean_gamma_deviance"""
    from CompStats.metrics import mean_gamma_deviance

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = mean_gamma_deviance(y_val,
                                 forest=hy,
                                 num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.mean_gamma_deviance(y_val, hy)
    assert _ == perf.statistic['forest']       


def test_mean_absolute_percentage_error():
    """mean_absolute_percentage_error"""
    from CompStats.metrics import mean_absolute_percentage_error

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = mean_absolute_percentage_error(y_val,
                                          forest=hy,
                                          num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.mean_absolute_percentage_error(y_val, hy)
    assert _ == perf.statistic['forest']


def test_d2_absolute_error_score():
    """d2_absolute_error_score"""
    from CompStats.metrics import d2_absolute_error_score

    X, y = load_diabetes(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestRegressor().fit(X_train, y_train)
    hy = ens.predict(X_val)
    perf = d2_absolute_error_score(y_val,
                                   forest=hy,
                                   num_samples=50)
    assert 'forest' in perf.statistic
    _ = metrics.d2_absolute_error_score(y_val, hy)
    assert _ == perf.statistic['forest']      