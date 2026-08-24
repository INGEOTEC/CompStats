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

import numpy as np
from sklearn.base import clone
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from CompStats.tests.test_performance import DATA


def test_Perf_name():
    """Test Perf name keyword"""
    from CompStats.metrics import f1_score
    score = f1_score([1, 0, 1], [1, 0, 0], name='algo')
    assert 'algo' in score.predictions


def test_Perf_plot_col_wrap():
    """Test plot when 2 classes"""
    from CompStats.metrics import f1_score

    X, y = load_breast_cancer(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    score = f1_score(y_val, ens.predict(X_val),
                     average=None,
                     num_samples=50)
    score(nb.predict(X_val))
    score(svm.predict(X_val))
    score.plot()


def test_Difference_dataframe():
    """Test Difference dataframe"""
    from CompStats.metrics import f1_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    score = f1_score(y_val, ens.predict(X_val),
                     average=None,
                     num_samples=50)
    score(nb.predict(X_val))
    score(svm.predict(X_val))
    diff = score.difference()
    df = diff.dataframe()
    assert 'Best' in df.columns
    score = f1_score(y_val, ens.predict(X_val),
                     average='macro',
                     num_samples=50)
    score(nb.predict(X_val))
    score(svm.predict(X_val))
    diff = score.difference()
    df = diff.dataframe()
    assert 'Best' not in df.columns


def test_Perf_dataframe():
    """Test Perf dataframe"""
    from CompStats.metrics import f1_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    score = f1_score(y_val, ens.predict(X_val),
                     average=None,
                     num_samples=50)
    df = score.dataframe()
    score(nb.predict(X_val))
    score(svm.predict(X_val))
    df = score.dataframe()
    assert 'Performance' in df.columns
    score = f1_score(y_val, ens.predict(X_val),
                     average='macro',
                     num_samples=50)
    score(nb.predict(X_val))
    score(svm.predict(X_val))
    df = score.dataframe()
    assert 'Performance' not in df.columns


def test_Perf_plot_multi():
    """Test Perf plot multiple"""
    from CompStats.metrics import f1_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    score = f1_score(y_val, ens.predict(X_val),
                     average=None,
                     num_samples=50)
    score(nb.predict(X_val))
    score(svm.predict(X_val))
    f_grid = score.plot()
    assert f_grid is not None


def test_Perf_statistic_one():
    """Test Perf statistic one alg"""
    from CompStats.metrics import f1_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    score = f1_score(y_val, ens.predict(X_val),
                     average=None,
                     num_samples=50)
    assert isinstance(score.statistic, np.ndarray)
    assert isinstance(str(score), str)
    score = f1_score(y_val, ens.predict(X_val),
                     average='macro',
                     num_samples=50)
    assert isinstance(score.statistic, float)
    assert isinstance(str(score), str)
    assert isinstance(score.se, float)
    assert isinstance(score.ci, tuple)
    assert len(score.ci) == 2


def test_Perf_best():
    """Test Perf best"""
    from CompStats.metrics import f1_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    score = f1_score(y_val, average=None,
                     num_samples=50)
    score(ens.predict(X_val), name='forest')
    score(nb.predict(X_val), name='NB')
    score(svm.predict(X_val), name='svm')
    assert isinstance(score.best, np.ndarray)
    score = f1_score(y_val, average='macro',
                     num_samples=50)
    score(ens.predict(X_val), name='forest')
    score(nb.predict(X_val), name='NB')
    score(svm.predict(X_val), name='svm')
    assert isinstance(score.best, str)


def test_difference_best():
    """Test multiple performance measures"""
    from CompStats.metrics import f1_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    score = f1_score(y_val, average=None,
                     num_samples=50)
    score(ens.predict(X_val), name='forest')
    score(nb.predict(X_val), name='NB')
    score(svm.predict(X_val), name='svm')
    diff = score.difference()
    assert isinstance(diff.best, np.ndarray)
    score = f1_score(y_val, average='macro',
                     num_samples=50)
    score(ens.predict(X_val), name='forest')
    score(nb.predict(X_val), name='NB')
    score(svm.predict(X_val), name='svm')
    diff = score.difference()
    assert isinstance(diff.best, str)


def test_difference_str__():
    """Test f1_score"""
    from CompStats.metrics import f1_score

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    perf = f1_score(y_val, nb.predict(X_val),
                    forest=ens.predict(X_val),
                    num_samples=50, average=None)
    diff = perf.difference()
    p_values = diff.p_value(right=False)
    dd = list(p_values.values())[0]
    assert isinstance(dd, np.ndarray)
    for average in ['macro', None]:
        perf = f1_score(y_val, nb.predict(X_val),
                        forest=ens.predict(X_val),
                        num_samples=50, average=average)
        diff = perf.difference()
        print(diff)


def test_Perf():
    """Test perf"""
    from CompStats.interface import Perf

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    m = LinearSVC().fit(X_train, y_train)
    hy = m.predict(X_val)
    ens = RandomForestClassifier().fit(X_train, y_train)
    perf = Perf(y_val, hy, forest=ens.predict(X_val), num_samples=50)
    assert 'alg-1' in perf.predictions
    assert 'forest' in perf.predictions
    assert str(perf) is not None


def test_Perf_statistic():
    """Test statistic"""
    from CompStats.interface import Perf

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    perf = Perf(y_val, forest=ens.predict(X_val), num_samples=50)
    perf(ens.predict(X_val))
    assert 'forest' in perf.statistic


def test_Perf_plot():
    """Test plot"""

    from CompStats.interface import Perf

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    perf = Perf(y_val, forest=ens.predict(X_val), num_samples=50)
    perf.plot()


def test_Perf_clone():
    """Test Perf.clone"""
    from CompStats.interface import Perf

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier().fit(X_train, y_train)
    perf = Perf(y_val, forest=ens.predict(X_val), num_samples=50)
    samples = perf.statistic_samples._samples
    perf2 = clone(perf)
    perf2.func = lambda y, hy: (y != hy).mean()
    perf2.BiB = False
    assert 'forest' in perf2.statistic_samples.calls
    assert np.all(samples == perf2.statistic_samples._samples)


def test_Perf_difference():
    """Test difference"""
    from CompStats.interface import Perf, Difference

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    m = LinearSVC().fit(X_train, y_train)
    hy = m.predict(X_val)
    ens = RandomForestClassifier().fit(X_train, y_train)
    perf = Perf(y_val, hy, forest=ens.predict(X_val), num_samples=50)
    diff = perf.difference()
    assert isinstance(diff, Difference)
    assert isinstance(str(diff), str)


def test_Difference_plot():
    """Test difference plot"""
    from CompStats.interface import Perf

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    m = LinearSVC().fit(X_train, y_train)
    hy = m.predict(X_val)
    ens = RandomForestClassifier().fit(X_train, y_train)
    perf = Perf(y_val, hy, forest=ens.predict(X_val), num_samples=50)
    diff = perf.difference()
    diff.plot()


def test_Perf_input_dataframe():
    """Test Perf with dataframe"""
    from CompStats.interface import Perf

    df = pd.read_csv(DATA)
    perf = Perf(df, num_samples=50)
    assert 'INGEOTEC' in perf.statistic


def test_Perf_multi_measure_score_only():
    """Test Perf combining two score-type measures via metrics.py's .measure() factories"""
    from CompStats.interface import Perf
    from CompStats.metrics import f1_score, recall_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    perf = Perf(y_val, ens.predict(X_val), nb=nb.predict(X_val),
                func=[f1_score.measure(average='macro'),
                      recall_score.measure(average='macro')],
                num_samples=20)
    assert perf.measure_names == ['f1_score', 'recall_score']
    assert isinstance(perf.statistic['alg-1'], np.ndarray)
    assert perf.statistic['alg-1'].shape == (2,)
    assert np.all(perf.statistic_samples.BiB == np.array([True, True]))
    df = perf.dataframe()
    assert set(df['Performance']) == {'f1_score', 'recall_score'}


def test_Perf_multi_measure_mixed_bib():
    """Test Perf/Difference with mixed score-type and error-type measures

    Uses constant prediction arrays so the bootstrap statistic has zero
    variance, making ``difference().p_value()`` exactly predictable; this
    isolates the per-column BiB sign logic (interface.py's difference()/
    p_value()/best) from sampling noise.
    """
    from CompStats.interface import Perf

    def score_stat(y, hy):
        return hy.mean()

    def error_stat(y, hy):
        return hy.mean()

    y_true = np.arange(10)
    hyA = np.full(10, 5.0)
    hyB = np.full(10, 2.0)
    perf = Perf(y_true, A=hyA, B=hyB,
                func=[score_stat, error_stat], BiB=[True, False],
                num_samples=5)
    assert perf.measure_names == ['score_stat', 'error_stat']
    assert np.all(perf.statistic_samples.BiB == np.array([True, False]))
    # A has the higher value (wins the score-type column),
    # B has the lower value (wins the error-type column)
    assert list(perf.best) == ['A', 'B']
    diff = perf.difference()
    p_values = diff.p_value()
    assert np.allclose(p_values['A'], [1.0, 0.0])
    assert np.allclose(p_values['B'], [0.0, 1.0])


def test_Perf_measure_tag_overrides_default_bib():
    """A callable's own .BiB (set by a .measure() factory) wins over the
    constructor's BiB default when the two disagree"""
    from CompStats.interface import Perf
    from CompStats.metrics import f1_score

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    hy = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    tagged = f1_score.measure(average='macro')
    assert tagged.BiB is True
    perf = Perf(y_true, alg=hy, func=tagged, BiB=False, num_samples=5)
    assert bool(perf.statistic_samples.BiB) is True


def test_Perf_statistic_sort_order_follows_tagged_bib():
    """Perf.statistic's sort order must follow the measure's tagged .BiB,
    not the constructor's BiB default (issue #36 regression: this used to
    be derived from ``score_func is not None`` and ignored the tag)"""
    from CompStats.interface import Perf
    from CompStats.metrics import mean_absolute_error

    y_true = np.zeros(10)
    perf = Perf(y_true, low=np.zeros(10), high=np.ones(10),
                func=mean_absolute_error.measure(), num_samples=5)
    # mean_absolute_error.measure() is tagged BiB=False (error-type) even
    # though the constructor's own BiB default is True; the smaller-error
    # 'low' prediction must rank first
    assert list(perf.statistic.keys()) == ['low', 'high']


def test_Perf_repr_label_uses_func():
    """Perf.__repr__ labels the measure as 'func=', regardless of its
    tagged direction (issue #36: score_func/error_func no longer exist)"""
    from CompStats.interface import Perf
    from CompStats.metrics import mean_absolute_error

    y_true = np.zeros(10)
    perf = Perf(y_true, low=np.zeros(10), high=np.ones(10),
                func=mean_absolute_error.measure(), num_samples=5)
    assert 'func=mean_absolute_error' in repr(perf)


def test_Perf_plot_value_name_follows_tagged_bib():
    """Perf.plot's default value_name label ('Score' vs 'Error') must follow
    the measure's tagged .BiB, not the constructor's BiB default (issue #36
    regression: this used to be derived from ``score_func is not None``)"""
    from CompStats.interface import Perf
    from CompStats.metrics import mean_absolute_error

    y_true = np.zeros(10)
    perf = Perf(y_true, low=np.zeros(10), high=np.ones(10),
                func=mean_absolute_error.measure(), num_samples=5)
    f_grid = perf.plot()
    assert 'Error' in f_grid.data.columns


def test_Perf_multi_measure_clone():
    """Test that cloning a multi-measure Perf preserves measures and samples"""
    from sklearn.base import clone
    from CompStats.interface import Perf
    from CompStats.metrics import f1_score, recall_score

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    perf = Perf(y_val, forest=ens.predict(X_val), nb=nb.predict(X_val),
                func=[f1_score.measure(average='macro'),
                      recall_score.measure(average='macro')],
                num_samples=20)
    samples = perf.statistic_samples._samples
    perf2 = clone(perf)
    assert perf2.measure_names == ['f1_score', 'recall_score']
    assert np.all(samples == perf2.statistic_samples._samples)
    assert np.allclose(perf.statistic['forest'], perf2.statistic['forest'])


def test_Perf_call():
    """Test Perf call"""
    from CompStats.interface import Perf

    X, y = load_iris(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = _
    m = LinearSVC().fit(X_train, y_train)
    hy = m.predict(X_val)
    ens = RandomForestClassifier().fit(X_train, y_train)
    hy2 = ens.predict(X_val)
    perf = Perf(y_val, num_samples=50)
    for xx in [hy, hy2]:
        _ = perf(xx)
        print(_)
    perf(hy, name='alg-2')
    assert 'alg-2' not in perf._statistic_samples.calls
    assert 'alg-1' in perf._statistic_samples.calls


def test_Difference_p_value_correction_single_measure():
    """Test Difference.p_value multiple-comparison correction (single measure)"""
    from statsmodels.stats.multitest import multipletests
    from CompStats.metrics import f1_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    score = f1_score(y_val, ens.predict(X_val), average='macro',
                     num_samples=50)
    score(nb.predict(X_val))
    score(svm.predict(X_val))
    diff = score.difference()
    raw = diff.p_value()
    keys = list(raw.keys())
    expected = multipletests(list(raw.values()), method='bonferroni')[1]
    corrected = diff.p_value(correction='bonferroni')
    assert list(corrected.keys()) == keys
    assert np.allclose(list(corrected.values()), expected)


def test_Difference_p_value_correction_multi_measure():
    """Test Difference.p_value multiple-comparison correction is applied per measure"""
    from statsmodels.stats.multitest import multipletests
    from CompStats.interface import Perf
    from CompStats.metrics import f1_score, recall_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    perf = Perf(y_val, ens.predict(X_val), nb=nb.predict(X_val),
                svm=svm.predict(X_val),
                func=[f1_score.measure(average='macro'),
                      recall_score.measure(average='macro')],
                num_samples=20)
    diff = perf.difference()
    raw = diff.p_value()
    corrected = diff.p_value(correction='bonferroni')
    keys = list(raw.keys())
    for col in range(2):
        expected = multipletests([raw[k][col] for k in keys],
                                 method='bonferroni')[1]
        actual = [corrected[k][col] for k in keys]
        assert np.allclose(actual, expected)
    for k in keys:
        assert np.all(corrected[k] >= raw[k] - 1e-12)


def test_Difference_dataframe_correction_changes_significant_flag():
    """Test that correcting p-values only makes Significant more conservative"""
    from CompStats.metrics import f1_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    score = f1_score(y_val, ens.predict(X_val), average='macro',
                     num_samples=50)
    score(nb.predict(X_val))
    score(svm.predict(X_val))
    diff = score.difference()
    n_significant = diff.dataframe()['Significant'].sum()
    n_significant_corrected = diff.dataframe(
        correction='bonferroni')['Significant'].sum()
    assert n_significant_corrected <= n_significant


def test_Perf_dataframe_correction_changes_comparison_legend():
    """Test that Perf.dataframe's Comparison legend reflects corrected p-values"""
    from CompStats.metrics import f1_score

    X, y = load_digits(return_X_y=True)
    _ = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train, X_val, y_train, y_val = _
    ens = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)
    score = f1_score(y_val, ens.predict(X_val), average='macro',
                     num_samples=50)
    score(nb.predict(X_val))
    score(svm.predict(X_val))
    uncorrected = score.dataframe(comparison=True)
    corrected = score.dataframe(comparison=True, correction='bonferroni')
    n_different = (uncorrected['Comparison'] == 'Different').sum()
    n_different_corrected = (corrected['Comparison'] == 'Different').sum()
    assert n_different_corrected <= n_different
