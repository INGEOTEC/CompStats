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

########################################################
#################### Classification ####################
########################################################

@perf_docs
def accuracy_score(y_true, *y_pred,
                   normalize=True, sample_weight=None,
                   num_samples: int=500,
                   n_jobs: int=-1, 
                   use_tqdm=True,
                   **kwargs):
    """accuracy_score"""

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
    """balanced_accuracy_score"""

    def inner(y, hy):
        return metrics.balanced_accuracy_score(y, hy,
                                               adjusted=adjusted,
                                               sample_weight=sample_weight)
    return Perf(y_true, *y_pred, score_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def top_k_accuracy_score(y_true, *y_score, k=2,
                         normalize=True, sample_weight=None,
                         labels=None,
                         num_samples: int=500,
                         n_jobs: int=-1,
                         use_tqdm=True,
                         **kwargs):
    """top_k_accuracy_score"""

    def inner(y, hy):
        return metrics.top_k_accuracy_score(y, hy, k=k,
                                            normalize=normalize, sample_weight=sample_weight,
                                            labels=labels)
    return Perf(y_true, *y_score, score_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def average_precision_score(y_true, *y_score,
                            average='macro',
                            sample_weight=None,
                            num_samples: int=500,
                            n_jobs: int=-1,
                            use_tqdm=True,
                            **kwargs):
    """average_precision_score"""

    def inner(y, hy):
        return metrics.average_precision_score(y, hy,
                                               average=average,
                                               sample_weight=sample_weight)
    return Perf(y_true, *y_score, score_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def brier_score_loss(y_true, *y_proba,
                     sample_weight=None,
                     pos_label=None,
                     num_samples: int=500,
                     n_jobs: int=-1,
                     use_tqdm=True,
                     **kwargs                     
                     ):
    """brier_score_loss"""

    def inner(y, hy):
        return metrics.brier_score_loss(y, hy,
                                        sample_weight=sample_weight,
                                        pos_label=pos_label)
    return Perf(y_true, *y_proba, score_func=None, error_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)
    

@perf_docs
def f1_score(y_true, *y_pred, labels=None, pos_label=1,
             average='binary', sample_weight=None,
             zero_division='warn', num_samples: int=500,
             n_jobs: int=-1, use_tqdm=True,
             **kwargs):
    """f1_score"""

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


@perf_docs
def log_loss(y_true, *y_pred,
             normalize=True,
             sample_weight=None,
             labels=None,
             num_samples: int=500,
             n_jobs: int=-1,
             use_tqdm=True,
             **kwargs):
    """log_loss"""
    def inner(y, hy):
        return metrics.log_loss(y, hy, normalize=normalize,
                                sample_weight=sample_weight,
                                labels=labels)
    return Perf(y_true, *y_pred, error_func=inner, score_func=None,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def precision_score(y_true,
                    *y_pred,
                    labels=None,
                    pos_label=1,
                    average='binary',
                    sample_weight=None,
                    zero_division='warn',
                    num_samples: int=500,
                    n_jobs: int=-1,
                    use_tqdm=True,
                    **kwargs):
    """precision_score"""
    def inner(y, hy):
        return metrics.precision_score(y, hy,
                                       labels=labels,
                                       pos_label=pos_label,
                                       average=average,
                                       sample_weight=sample_weight,
                                       zero_division=zero_division)
    return Perf(y_true, *y_pred, score_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def recall_score(y_true,
                 *y_pred,
                 labels=None,
                 pos_label=1,
                 average='binary',
                 sample_weight=None,
                 zero_division='warn',
                 num_samples: int=500,
                 n_jobs: int=-1,
                 use_tqdm=True,
                 **kwargs):
    """recall_score"""
    def inner(y, hy):
        return metrics.recall_score(y, hy,
                                    labels=labels,
                                    pos_label=pos_label,
                                    average=average,
                                    sample_weight=sample_weight,
                                    zero_division=zero_division)
    return Perf(y_true, *y_pred, score_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def jaccard_score():
    """jaccard_score"""
    raise RuntimeError('Not Implemented')


@perf_docs
def roc_auc_score():
    """roc_auc_score"""
    raise RuntimeError('Not Implemented')


@perf_docs
def d2_log_loss_score():
    """d2_log_loss_score"""
    raise RuntimeError('Not Implemented')


########################################################
#################### Regression ########################
########################################################


@perf_docs
def explained_variance_score(y_true,
                             *y_pred,
                             sample_weight=None,
                             multioutput='uniform_average',
                             force_finite=True,
                             num_samples: int=500,
                             n_jobs: int=-1,
                             use_tqdm=True,
                             **kwargs):
    """explained_variance_score"""
    def inner(y, hy):
        return metrics.explained_variance_score(y, hy,
                                                sample_weight=sample_weight,
                                                multioutput=multioutput,
                                                force_finite=force_finite)
    return Perf(y_true, *y_pred, score_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def max_error(y_true, *y_pred, 
              num_samples: int=500,
              n_jobs: int=-1,
              use_tqdm=True,
              **kwargs):
    """max_error"""
    def inner(y, hy):
        return metrics.max_error(y, hy)
    return Perf(y_true, *y_pred, score_func=None, error_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def mean_absolute_error(y_true,
                        *y_pred,
                        sample_weight=None,
                        multioutput='uniform_average',
                        num_samples: int=500,
                        n_jobs: int=-1,
                        use_tqdm=True,
                        **kwargs):
    """mean_absolute_error"""
    def inner(y, hy):
        return metrics.mean_absolute_error(y, hy,
                                           sample_weight=sample_weight,
                                           multioutput=multioutput)

    return Perf(y_true, *y_pred, score_func=None, error_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def mean_squared_error(y_true,
                       *y_pred,
                       sample_weight=None,
                       multioutput='uniform_average',
                       num_samples: int=500,
                       n_jobs: int=-1,
                       use_tqdm=True,
                       **kwargs):
    """mean_squared_error"""
    def inner(y, hy):
        return metrics.mean_squared_error(y, hy,
                                          sample_weight=sample_weight,
                                          multioutput=multioutput)

    return Perf(y_true, *y_pred, score_func=None, error_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def root_mean_squared_error(y_true,
                            *y_pred,
                            sample_weight=None,
                            multioutput='uniform_average',
                            num_samples: int=500,
                            n_jobs: int=-1,
                            use_tqdm=True,
                            **kwargs):
    """root_mean_squared_error"""
    def inner(y, hy):
        return metrics.root_mean_squared_error(y, hy,
                                               sample_weight=sample_weight,
                                               multioutput=multioutput)

    return Perf(y_true, *y_pred, score_func=None, error_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def mean_squared_log_error(y_true,
                           *y_pred,
                           sample_weight=None,
                           multioutput='uniform_average',
                           num_samples: int=500,
                           n_jobs: int=-1,
                           use_tqdm=True,
                           **kwargs):
    """mean_squared_log_error"""
    def inner(y, hy):
        return metrics.mean_squared_log_error(y, hy,
                                              sample_weight=sample_weight,
                                              multioutput=multioutput)

    return Perf(y_true, *y_pred, score_func=None, error_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)


@perf_docs
def root_mean_squared_log_error(y_true,
                                *y_pred,
                                sample_weight=None,
                                multioutput='uniform_average',
                                num_samples: int=500,
                                n_jobs: int=-1,
                                use_tqdm=True,
                                **kwargs):
    """root_mean_squared_log_error"""
    def inner(y, hy):
        return metrics.root_mean_squared_log_error(y, hy,
                                                   sample_weight=sample_weight,
                                                   multioutput=multioutput)

    return Perf(y_true, *y_pred, score_func=None, error_func=inner,
                num_samples=num_samples, n_jobs=n_jobs,
                use_tqdm=use_tqdm,
                **kwargs)
