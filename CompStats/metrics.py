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
from functools import wraps
from sklearn import metrics
from scipy import stats
from CompStats.interface import Perf
from CompStats.utils import metrics_docs


########################################################
#################### Classification ####################
########################################################


def _accuracy_score_measure(normalize=True, sample_weight=None):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`accuracy_score`"""

    @wraps(metrics.accuracy_score)
    def inner(y, hy):
        return metrics.accuracy_score(y, hy,
                                      normalize=normalize,
                                      sample_weight=sample_weight)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_pred', bib=True)
def accuracy_score(y_true, *y_pred,
                   normalize=True, sample_weight=None,
                   num_samples: int = 500,
                   n_jobs: int = -1,
                   use_tqdm=True,
                   **kwargs):
    """accuracy_score"""

    return Perf(y_true, *y_pred,
                func=_accuracy_score_measure(normalize=normalize,
                                             sample_weight=sample_weight),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


accuracy_score.measure = _accuracy_score_measure


def _balanced_accuracy_score_measure(sample_weight=None, adjusted=False):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`balanced_accuracy_score`"""

    @wraps(metrics.balanced_accuracy_score)
    def inner(y, hy):
        return metrics.balanced_accuracy_score(y, hy,
                                               adjusted=adjusted,
                                               sample_weight=sample_weight)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_pred', bib=True)
def balanced_accuracy_score(y_true, *y_pred,
                            sample_weight=None, adjusted=False,
                            num_samples: int = 500,
                            n_jobs: int = -1,
                            use_tqdm=True,
                            **kwargs):
    """balanced_accuracy_score"""

    return Perf(y_true, *y_pred,
                func=_balanced_accuracy_score_measure(sample_weight=sample_weight,
                                                      adjusted=adjusted),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


balanced_accuracy_score.measure = _balanced_accuracy_score_measure


def _top_k_accuracy_score_measure(k=2, normalize=True, sample_weight=None, labels=None):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`top_k_accuracy_score`"""

    @wraps(metrics.top_k_accuracy_score)
    def inner(y, hy):
        return metrics.top_k_accuracy_score(y, hy, k=k,
                                            normalize=normalize, sample_weight=sample_weight,
                                            labels=labels)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_score', bib=True)
def top_k_accuracy_score(y_true, *y_score, k=2,
                         normalize=True, sample_weight=None,
                         labels=None,
                         num_samples: int = 500,
                         n_jobs: int = -1,
                         use_tqdm=True,
                         **kwargs):
    """top_k_accuracy_score"""

    return Perf(y_true, *y_score,
                func=_top_k_accuracy_score_measure(k=k, normalize=normalize,
                                                   sample_weight=sample_weight,
                                                   labels=labels),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


top_k_accuracy_score.measure = _top_k_accuracy_score_measure


def _average_precision_score_measure(average='macro', sample_weight=None):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`average_precision_score`"""

    @wraps(metrics.average_precision_score)
    def inner(y, hy):
        return metrics.average_precision_score(y, hy,
                                               average=average,
                                               sample_weight=sample_weight)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_score', bib=True)
def average_precision_score(y_true, *y_score,
                            average='macro',
                            sample_weight=None,
                            num_samples: int = 500,
                            n_jobs: int = -1,
                            use_tqdm=True,
                            **kwargs):
    """average_precision_score"""

    return Perf(y_true, *y_score,
                func=_average_precision_score_measure(average=average,
                                                      sample_weight=sample_weight),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


average_precision_score.measure = _average_precision_score_measure


def _brier_score_loss_measure(sample_weight=None, pos_label=None):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`brier_score_loss`"""

    @wraps(metrics.brier_score_loss)
    def inner(y, hy):
        return metrics.brier_score_loss(y, hy,
                                        sample_weight=sample_weight,
                                        pos_label=pos_label)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_proba', bib=False)
def brier_score_loss(y_true, *y_proba,
                     sample_weight=None,
                     pos_label=None,
                     num_samples: int = 500,
                     n_jobs: int = -1,
                     use_tqdm=True,
                     **kwargs
                     ):
    """brier_score_loss"""

    return Perf(y_true, *y_proba,
                func=_brier_score_loss_measure(sample_weight=sample_weight,
                                               pos_label=pos_label),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


brier_score_loss.measure = _brier_score_loss_measure


def _f1_score_measure(labels=None, pos_label=1, average='binary',
                      sample_weight=None, zero_division='warn'):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`f1_score`"""

    @wraps(metrics.f1_score)
    def inner(y, hy):
        return metrics.f1_score(y, hy, labels=labels,
                                pos_label=pos_label,
                                average=average,
                                sample_weight=sample_weight,
                                zero_division=zero_division)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_pred', bib=True)
def f1_score(y_true, *y_pred, labels=None, pos_label=1,
             average='binary', sample_weight=None,
             zero_division='warn', num_samples: int = 500,
             n_jobs: int = -1, use_tqdm=True,
             **kwargs):
    """f1_score"""

    return Perf(y_true, *y_pred,
                func=_f1_score_measure(labels=labels, pos_label=pos_label,
                                       average=average,
                                       sample_weight=sample_weight,
                                       zero_division=zero_division),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


f1_score.measure = _f1_score_measure


def _log_loss_measure(normalize=True, sample_weight=None, labels=None):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`log_loss`"""

    @wraps(metrics.log_loss)
    def inner(y, hy):
        return metrics.log_loss(y, hy, normalize=normalize,
                                sample_weight=sample_weight,
                                labels=labels)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def log_loss(y_true, *y_pred,
             normalize=True,
             sample_weight=None,
             labels=None,
             num_samples: int = 500,
             n_jobs: int = -1,
             use_tqdm=True,
             **kwargs):
    """log_loss"""

    return Perf(y_true, *y_pred,
                func=_log_loss_measure(normalize=normalize,
                                       sample_weight=sample_weight,
                                       labels=labels),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


log_loss.measure = _log_loss_measure


def _precision_score_measure(labels=None, pos_label=1, average='binary',
                             sample_weight=None, zero_division='warn'):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`precision_score`"""

    @wraps(metrics.precision_score)
    def inner(y, hy):
        return metrics.precision_score(y, hy,
                                       labels=labels,
                                       pos_label=pos_label,
                                       average=average,
                                       sample_weight=sample_weight,
                                       zero_division=zero_division)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_pred', bib=True)
def precision_score(y_true,
                    *y_pred,
                    labels=None,
                    pos_label=1,
                    average='binary',
                    sample_weight=None,
                    zero_division='warn',
                    num_samples: int = 500,
                    n_jobs: int = -1,
                    use_tqdm=True,
                    **kwargs):
    """precision_score"""

    return Perf(y_true, *y_pred,
                func=_precision_score_measure(labels=labels, pos_label=pos_label,
                                              average=average,
                                              sample_weight=sample_weight,
                                              zero_division=zero_division),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


precision_score.measure = _precision_score_measure


def _recall_score_measure(labels=None, pos_label=1, average='binary',
                          sample_weight=None, zero_division='warn'):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`recall_score`"""

    @wraps(metrics.recall_score)
    def inner(y, hy):
        return metrics.recall_score(y, hy,
                                    labels=labels,
                                    pos_label=pos_label,
                                    average=average,
                                    sample_weight=sample_weight,
                                    zero_division=zero_division)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_pred', bib=True)
def recall_score(y_true,
                 *y_pred,
                 labels=None,
                 pos_label=1,
                 average='binary',
                 sample_weight=None,
                 zero_division='warn',
                 num_samples: int = 500,
                 n_jobs: int = -1,
                 use_tqdm=True,
                 **kwargs):
    """recall_score"""

    return Perf(y_true, *y_pred,
                func=_recall_score_measure(labels=labels, pos_label=pos_label,
                                           average=average,
                                           sample_weight=sample_weight,
                                           zero_division=zero_division),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


recall_score.measure = _recall_score_measure


def _jaccard_score_measure(labels=None, pos_label=1, average='binary',
                           sample_weight=None, zero_division='warn'):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`jaccard_score`"""

    @wraps(metrics.jaccard_score)
    def inner(y, hy):
        return metrics.jaccard_score(y, hy,
                                     labels=labels,
                                     pos_label=pos_label,
                                     average=average,
                                     sample_weight=sample_weight,
                                     zero_division=zero_division)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_pred', bib=True)
def jaccard_score(y_true,
                  *y_pred,
                  labels=None,
                  pos_label=1,
                  average='binary',
                  sample_weight=None,
                  zero_division='warn',
                  num_samples: int = 500,
                  n_jobs: int = -1,
                  use_tqdm=True,
                  **kwargs):
    """jaccard_score"""

    return Perf(y_true, *y_pred,
                func=_jaccard_score_measure(labels=labels, pos_label=pos_label,
                                            average=average,
                                            sample_weight=sample_weight,
                                            zero_division=zero_division),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


jaccard_score.measure = _jaccard_score_measure


def _roc_auc_score_measure(average='macro', sample_weight=None, max_fpr=None,
                           multi_class='raise', labels=None):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`roc_auc_score`"""

    @wraps(metrics.roc_auc_score)
    def inner(y, hy):
        return metrics.roc_auc_score(y, hy,
                                     average=average,
                                     sample_weight=sample_weight,
                                     max_fpr=max_fpr,
                                     multi_class=multi_class,
                                     labels=labels)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_score', bib=True)
def roc_auc_score(y_true,
                  *y_score,
                  average='macro',
                  sample_weight=None,
                  max_fpr=None,
                  multi_class='raise',
                  labels=None,
                  num_samples: int = 500,
                  n_jobs: int = -1,
                  use_tqdm=True,
                  **kwargs):
    """roc_auc_score"""

    return Perf(y_true, *y_score,
                func=_roc_auc_score_measure(average=average,
                                            sample_weight=sample_weight,
                                            max_fpr=max_fpr,
                                            multi_class=multi_class,
                                            labels=labels),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


roc_auc_score.measure = _roc_auc_score_measure


def _d2_log_loss_score_measure(sample_weight=None, labels=None):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`d2_log_loss_score`"""

    @wraps(metrics.d2_log_loss_score)
    def inner(y, hy):
        return metrics.d2_log_loss_score(y, hy,
                                         sample_weight=sample_weight,
                                         labels=labels)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_proba', bib=True)
def d2_log_loss_score(y_true, *y_proba,
                      sample_weight=None,
                      labels=None,
                      num_samples: int = 500,
                      n_jobs: int = -1,
                      use_tqdm=True,
                      **kwargs):
    """d2_log_loss_score"""

    return Perf(y_true, *y_proba,
                func=_d2_log_loss_score_measure(sample_weight=sample_weight,
                                                labels=labels),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


d2_log_loss_score.measure = _d2_log_loss_score_measure


def _macro_f1_measure(labels=None, sample_weight=None, zero_division='warn'):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`macro_f1`"""

    return f1_score.measure(labels=labels, average='macro',
                            sample_weight=sample_weight,
                            zero_division=zero_division)


def macro_f1(y_true, *y_pred, labels=None,
             sample_weight=None, zero_division='warn',
             num_samples: int = 500, n_jobs: int = -1, use_tqdm=True,
             **kwargs):
    """:py:class:`~CompStats.interface.Perf` with :py:func:`~sklearn.metrics.f1_score` (as :py:attr:`func`) with the parameteres needed to compute the macro score. The parameters not described can be found in :py:func:`~sklearn.metrics.f1_score`

    :param y_true: True measurement or could be a pandas.DataFrame where column label 'y' corresponds to the true measurement.
    :type y_true: numpy.ndarray or pandas.DataFrame
    :param y_pred: Predictions, the algorithms will be identified with alg-k where k=1 is the first argument included in :py:attr:`y_pred.`
    :type y_pred: numpy.ndarray
    :param kwargs: Predictions, the algorithms will be identified using the keyword
    :type kwargs: numpy.ndarray
    :param num_samples: Number of bootstrap samples, default=500.
    :type num_samples: int
    :param n_jobs: Number of jobs to compute the statistic, default=-1 corresponding to use all threads.
    :type n_jobs: int
    :param use_tqdm: Whether to use tqdm.tqdm to visualize the progress, default=True
    :type use_tqdm: bool
    """
    return f1_score(y_true, *y_pred, labels=labels, average='macro',
                    sample_weight=sample_weight, zero_division=zero_division,
                    num_samples=num_samples, n_jobs=n_jobs,
                    use_tqdm=use_tqdm, **kwargs)


macro_f1.measure = _macro_f1_measure


def _macro_recall_measure(labels=None, sample_weight=None, zero_division='warn'):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`macro_recall`"""

    return recall_score.measure(labels=labels, average='macro',
                                sample_weight=sample_weight,
                                zero_division=zero_division)


def macro_recall(y_true, *y_pred, labels=None,
                 sample_weight=None, zero_division='warn',
                 num_samples: int = 500, n_jobs: int = -1, use_tqdm=True,
                 **kwargs):
    """:py:class:`~CompStats.interface.Perf` with :py:func:`~sklearn.metrics.recall_score` (as :py:attr:`func`) with the parameteres needed to compute the macro score. The parameters not described can be found in :py:func:`~sklearn.metrics.recall_score`

    :param y_true: True measurement or could be a pandas.DataFrame where column label 'y' corresponds to the true measurement.
    :type y_true: numpy.ndarray or pandas.DataFrame
    :param y_pred: Predictions, the algorithms will be identified with alg-k where k=1 is the first argument included in :py:attr:`y_pred.`
    :type y_pred: numpy.ndarray
    :param kwargs: Predictions, the algorithms will be identified using the keyword
    :type kwargs: numpy.ndarray
    :param num_samples: Number of bootstrap samples, default=500.
    :type num_samples: int
    :param n_jobs: Number of jobs to compute the statistic, default=-1 corresponding to use all threads.
    :type n_jobs: int
    :param use_tqdm: Whether to use tqdm.tqdm to visualize the progress, default=True
    :type use_tqdm: bool
    """
    return recall_score(y_true, *y_pred, labels=labels, average='macro',
                        sample_weight=sample_weight, zero_division=zero_division,
                        num_samples=num_samples, n_jobs=n_jobs,
                        use_tqdm=use_tqdm, **kwargs)


macro_recall.measure = _macro_recall_measure


def _macro_precision_measure(labels=None, sample_weight=None, zero_division='warn'):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`macro_precision`"""

    return precision_score.measure(labels=labels, average='macro',
                                   sample_weight=sample_weight,
                                   zero_division=zero_division)


def macro_precision(y_true, *y_pred, labels=None,
                    sample_weight=None, zero_division='warn',
                    num_samples: int = 500, n_jobs: int = -1, use_tqdm=True,
                    **kwargs):
    """:py:class:`~CompStats.interface.Perf` with :py:func:`~sklearn.metrics.precision_score` (as :py:attr:`func`) with the parameteres needed to compute the macro score. The parameters not described can be found in :py:func:`~sklearn.metrics.precision_score`

    :param y_true: True measurement or could be a pandas.DataFrame where column label 'y' corresponds to the true measurement.
    :type y_true: numpy.ndarray or pandas.DataFrame
    :param y_pred: Predictions, the algorithms will be identified with alg-k where k=1 is the first argument included in :py:attr:`y_pred.`
    :type y_pred: numpy.ndarray
    :param kwargs: Predictions, the algorithms will be identified using the keyword
    :type kwargs: numpy.ndarray
    :param num_samples: Number of bootstrap samples, default=500.
    :type num_samples: int
    :param n_jobs: Number of jobs to compute the statistic, default=-1 corresponding to use all threads.
    :type n_jobs: int
    :param use_tqdm: Whether to use tqdm.tqdm to visualize the progress, default=True
    :type use_tqdm: bool
    """
    return precision_score(y_true, *y_pred, labels=labels, average='macro',
                           sample_weight=sample_weight, zero_division=zero_division,
                           num_samples=num_samples, n_jobs=n_jobs,
                           use_tqdm=use_tqdm, **kwargs)


macro_precision.measure = _macro_precision_measure


########################################################
#################### Regression ########################
########################################################


def _explained_variance_score_measure(sample_weight=None, multioutput='uniform_average',
                                      force_finite=True):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`explained_variance_score`"""

    @wraps(metrics.explained_variance_score)
    def inner(y, hy):
        return metrics.explained_variance_score(y, hy,
                                                sample_weight=sample_weight,
                                                multioutput=multioutput,
                                                force_finite=force_finite)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_pred', bib=True)
def explained_variance_score(y_true,
                             *y_pred,
                             sample_weight=None,
                             multioutput='uniform_average',
                             force_finite=True,
                             num_samples: int = 500,
                             n_jobs: int = -1,
                             use_tqdm=True,
                             **kwargs):
    """explained_variance_score"""

    return Perf(y_true, *y_pred,
                func=_explained_variance_score_measure(sample_weight=sample_weight,
                                                       multioutput=multioutput,
                                                       force_finite=force_finite),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


explained_variance_score.measure = _explained_variance_score_measure


def _max_error_measure():
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`max_error`"""

    @wraps(metrics.max_error)
    def inner(y, hy):
        return metrics.max_error(y, hy)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def max_error(y_true, *y_pred,
              num_samples: int = 500,
              n_jobs: int = -1,
              use_tqdm=True,
              **kwargs):
    """max_error"""

    return Perf(y_true, *y_pred,
                func=_max_error_measure(),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


max_error.measure = _max_error_measure


def _mean_absolute_error_measure(sample_weight=None, multioutput='uniform_average'):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`mean_absolute_error`"""

    @wraps(metrics.mean_absolute_error)
    def inner(y, hy):
        return metrics.mean_absolute_error(y, hy,
                                           sample_weight=sample_weight,
                                           multioutput=multioutput)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def mean_absolute_error(y_true,
                        *y_pred,
                        sample_weight=None,
                        multioutput='uniform_average',
                        num_samples: int = 500,
                        n_jobs: int = -1,
                        use_tqdm=True,
                        **kwargs):
    """mean_absolute_error"""

    return Perf(y_true, *y_pred,
                func=_mean_absolute_error_measure(sample_weight=sample_weight,
                                                  multioutput=multioutput),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


mean_absolute_error.measure = _mean_absolute_error_measure


def _mean_squared_error_measure(sample_weight=None, multioutput='uniform_average'):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`mean_squared_error`"""

    @wraps(metrics.mean_squared_error)
    def inner(y, hy):
        return metrics.mean_squared_error(y, hy,
                                          sample_weight=sample_weight,
                                          multioutput=multioutput)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def mean_squared_error(y_true,
                       *y_pred,
                       sample_weight=None,
                       multioutput='uniform_average',
                       num_samples: int = 500,
                       n_jobs: int = -1,
                       use_tqdm=True,
                       **kwargs):
    """mean_squared_error"""

    return Perf(y_true, *y_pred,
                func=_mean_squared_error_measure(sample_weight=sample_weight,
                                                 multioutput=multioutput),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


mean_squared_error.measure = _mean_squared_error_measure


def _root_mean_squared_error_measure(sample_weight=None, multioutput='uniform_average'):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`root_mean_squared_error`"""

    @wraps(metrics.root_mean_squared_error)
    def inner(y, hy):
        return metrics.root_mean_squared_error(y, hy,
                                               sample_weight=sample_weight,
                                               multioutput=multioutput)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def root_mean_squared_error(y_true,
                            *y_pred,
                            sample_weight=None,
                            multioutput='uniform_average',
                            num_samples: int = 500,
                            n_jobs: int = -1,
                            use_tqdm=True,
                            **kwargs):
    """root_mean_squared_error"""

    return Perf(y_true, *y_pred,
                func=_root_mean_squared_error_measure(sample_weight=sample_weight,
                                                      multioutput=multioutput),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


root_mean_squared_error.measure = _root_mean_squared_error_measure


def _mean_squared_log_error_measure(sample_weight=None, multioutput='uniform_average'):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`mean_squared_log_error`"""

    @wraps(metrics.mean_squared_log_error)
    def inner(y, hy):
        return metrics.mean_squared_log_error(y, hy,
                                              sample_weight=sample_weight,
                                              multioutput=multioutput)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def mean_squared_log_error(y_true,
                           *y_pred,
                           sample_weight=None,
                           multioutput='uniform_average',
                           num_samples: int = 500,
                           n_jobs: int = -1,
                           use_tqdm=True,
                           **kwargs):
    """mean_squared_log_error"""

    return Perf(y_true, *y_pred,
                func=_mean_squared_log_error_measure(sample_weight=sample_weight,
                                                     multioutput=multioutput),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


mean_squared_log_error.measure = _mean_squared_log_error_measure


def _root_mean_squared_log_error_measure(sample_weight=None, multioutput='uniform_average'):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`root_mean_squared_log_error`"""

    @wraps(metrics.root_mean_squared_log_error)
    def inner(y, hy):
        return metrics.root_mean_squared_log_error(y, hy,
                                                   sample_weight=sample_weight,
                                                   multioutput=multioutput)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def root_mean_squared_log_error(y_true,
                                *y_pred,
                                sample_weight=None,
                                multioutput='uniform_average',
                                num_samples: int = 500,
                                n_jobs: int = -1,
                                use_tqdm=True,
                                **kwargs):
    """root_mean_squared_log_error"""

    return Perf(y_true, *y_pred,
                func=_root_mean_squared_log_error_measure(sample_weight=sample_weight,
                                                          multioutput=multioutput),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


root_mean_squared_log_error.measure = _root_mean_squared_log_error_measure


def _median_absolute_error_measure(sample_weight=None, multioutput='uniform_average'):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`median_absolute_error`"""

    @wraps(metrics.median_absolute_error)
    def inner(y, hy):
        return metrics.median_absolute_error(y, hy,
                                             sample_weight=sample_weight,
                                             multioutput=multioutput)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def median_absolute_error(y_true,
                          *y_pred,
                          sample_weight=None,
                          multioutput='uniform_average',
                          num_samples: int = 500,
                          n_jobs: int = -1,
                          use_tqdm=True,
                          **kwargs):
    """median_absolute_error"""

    return Perf(y_true, *y_pred,
                func=_median_absolute_error_measure(sample_weight=sample_weight,
                                                    multioutput=multioutput),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


median_absolute_error.measure = _median_absolute_error_measure


def _r2_score_measure(sample_weight=None, multioutput='uniform_average', force_finite=True):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`r2_score`"""

    @wraps(metrics.r2_score)
    def inner(y, hy):
        return metrics.r2_score(y, hy,
                                sample_weight=sample_weight,
                                multioutput=multioutput,
                                force_finite=force_finite)
    inner.BiB = True
    return inner


@metrics_docs(hy_name='y_pred', bib=True)
def r2_score(y_true,
             *y_pred,
             sample_weight=None,
             multioutput='uniform_average',
             force_finite=True,
             num_samples: int = 500,
             n_jobs: int = -1,
             use_tqdm=True,
             **kwargs):
    """r2_score"""

    return Perf(y_true, *y_pred,
                func=_r2_score_measure(sample_weight=sample_weight,
                                       multioutput=multioutput,
                                       force_finite=force_finite),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


r2_score.measure = _r2_score_measure


def _mean_poisson_deviance_measure(sample_weight=None):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`mean_poisson_deviance`"""

    @wraps(metrics.mean_poisson_deviance)
    def inner(y, hy):
        return metrics.mean_poisson_deviance(y, hy,
                                             sample_weight=sample_weight)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def mean_poisson_deviance(y_true,
                          *y_pred,
                          sample_weight=None,
                          num_samples: int = 500,
                          n_jobs: int = -1,
                          use_tqdm=True,
                          **kwargs):
    """mean_poisson_deviance"""

    return Perf(y_true, *y_pred,
                func=_mean_poisson_deviance_measure(sample_weight=sample_weight),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


mean_poisson_deviance.measure = _mean_poisson_deviance_measure


def _mean_gamma_deviance_measure(sample_weight=None):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`mean_gamma_deviance`"""

    @wraps(metrics.mean_gamma_deviance)
    def inner(y, hy):
        return metrics.mean_gamma_deviance(y, hy,
                                           sample_weight=sample_weight)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def mean_gamma_deviance(y_true,
                        *y_pred,
                        sample_weight=None,
                        num_samples: int = 500,
                        n_jobs: int = -1,
                        use_tqdm=True,
                        **kwargs):
    """mean_gamma_deviance"""

    return Perf(y_true, *y_pred,
                func=_mean_gamma_deviance_measure(sample_weight=sample_weight),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


mean_gamma_deviance.measure = _mean_gamma_deviance_measure


def _mean_absolute_percentage_error_measure(sample_weight=None, multioutput='uniform_average'):
    """Build the tagged (error-type, BiB=False) measure used by :py:func:`mean_absolute_percentage_error`"""

    @wraps(metrics.mean_absolute_percentage_error)
    def inner(y, hy):
        return metrics.mean_absolute_percentage_error(y, hy,
                                                      sample_weight=sample_weight,
                                                      multioutput=multioutput)
    inner.BiB = False
    return inner


@metrics_docs(hy_name='y_pred', bib=False)
def mean_absolute_percentage_error(y_true,
                                   *y_pred,
                                   sample_weight=None,
                                   multioutput='uniform_average',
                                   num_samples: int = 500,
                                   n_jobs: int = -1,
                                   use_tqdm=True,
                                   **kwargs):
    """mean_absolute_percentage_error"""

    return Perf(y_true, *y_pred,
                func=_mean_absolute_percentage_error_measure(sample_weight=sample_weight,
                                                             multioutput=multioutput),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


mean_absolute_percentage_error.measure = _mean_absolute_percentage_error_measure


def _d2_absolute_error_score_measure(sample_weight=None, multioutput='uniform_average'):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`d2_absolute_error_score`"""

    @wraps(metrics.d2_absolute_error_score)
    def inner(y, hy):
        return metrics.d2_absolute_error_score(y, hy,
                                               sample_weight=sample_weight,
                                               multioutput=multioutput)
    inner.BiB = True
    return inner


def d2_absolute_error_score(y_true,
                            *y_pred,
                            sample_weight=None,
                            multioutput='uniform_average',
                            num_samples: int = 500,
                            n_jobs: int = -1,
                            use_tqdm=True,
                            **kwargs):
    """d2_absolute_error_score"""

    return Perf(y_true, *y_pred,
                func=_d2_absolute_error_score_measure(sample_weight=sample_weight,
                                                      multioutput=multioutput),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


d2_absolute_error_score.measure = _d2_absolute_error_score_measure


def _pearsonr_measure(alternative='two-sided', method=None):
    """Build the tagged (score-type, BiB=True) measure used by :py:func:`pearsonr`"""

    @wraps(stats.pearsonr)
    def inner(y, hy):
        return stats.pearsonr(y, hy,
                              alternative=alternative,
                              method=method).statistic
    inner.BiB = True
    return inner


def pearsonr(y_true, *y_pred,
             alternative='two-sided', method=None,
             num_samples: int = 500,
             n_jobs: int = -1,
             use_tqdm=True,
             **kwargs):
    """:py:class:`~CompStats.interface.Perf` with :py:func:`~scipy.stats.pearsonr` as :py:attr:`func.`

    :param y_true: True measurement or could be a pandas.DataFrame where column label 'y' corresponds to the true measurement.
    :type y_true: numpy.ndarray or pandas.DataFrame
    :param y_pred: Predictions, the algorithms will be identified with alg-k where k=1 is the first argument included in :py:attr:`y_pred.`
    :type y_pred: numpy.ndarray
    :param kwargs: Predictions, the algorithms will be identified using the keyword
    :type kwargs: numpy.ndarray
    :param num_samples: Number of bootstrap samples, default=500.
    :type num_samples: int
    :param n_jobs: Number of jobs to compute the statistic, default=-1 corresponding to use all threads.
    :type n_jobs: int
    :param use_tqdm: Whether to use tqdm.tqdm to visualize the progress, default=True
    :type use_tqdm: bool
    """

    return Perf(y_true, *y_pred,
                func=_pearsonr_measure(alternative=alternative, method=method),
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


pearsonr.measure = _pearsonr_measure
