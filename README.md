# CompStats

[![Tests](https://github.com/INGEOTEC/CompStats/actions/workflows/test.yml/badge.svg)](https://github.com/INGEOTEC/CompStats/actions/workflows/test.yml)

[![Coverage](https://coveralls.io/repos/github/INGEOTEC/CompStats/badge.svg?branch=develop)](https://coveralls.io/github/INGEOTEC/CompStats?branch=develop)

[![PyPI version](https://badge.fury.io/py/CompStats.svg)](https://badge.fury.io/py/CompStats)

[![conda-forge build](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/compstats-feedstock?branchName=main)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=20297&branchName=main)

[![conda-forge version](https://img.shields.io/conda/vn/conda-forge/compstats.svg)](https://anaconda.org/conda-forge/compstats)

[![conda-forge platforms](https://img.shields.io/conda/pn/conda-forge/compstats.svg)](https://anaconda.org/conda-forge/compstats)

CompStats is an open-source Python package for the statistical analysis of the results of
supervised-learning systems evaluated on a shared held-out test set, the setting of collaborative
competitions (also called challenges). From the gold standard and the predictions of each system,
it estimates by bootstrap the standard error and the confidence interval of every performance
measure, tests whether the difference between the best system and each of the others is
statistically significant, and corrects those tests for multiple comparisons. It implements the
methodology of the article [Analysis of systems' performance in natural language processing
competitions](https://doi.org/10.1016/j.patrec.2024.03.010) (Pattern Recognition Letters, 2024),
which complements the standard selection of a winner, usually based only on the ranking of the
scores.

## Documentation

- Using the package: https://ingeotec.github.io/CompStats/
- Extending the package (developer reference): https://ingeotec.github.io/CompStats/dev/

## Installation

The package is installed from PyPI with

```bash
pip install CompStats
```

or from conda-forge with

```bash
conda install --channel conda-forge CompStats
```

## Quickstart

The following example compares three classifiers on the digits problem.

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from CompStats.metrics import f1_score

X, y = load_digits(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

svm = LinearSVC(random_state=0).fit(X_train, y_train)
score = f1_score(y_val, svm.predict(X_val), average='macro', name='Linear SVM')
score(RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_val),
      name='Random Forest')
score(GaussianNB().fit(X_train, y_train).predict(X_val), name='Naive Bayes')

score                 # point estimate and standard error of every system
score.ci              # 95% percentile bootstrap confidence intervals
diff = score.difference()            # paired differences against the best system
diff.p_value(correction='holm')      # p-values adjusted for multiple comparisons
score.plot()                         # performance with confidence intervals
```

The variable `score` holds the point estimate and the standard error of every system, and
`score.ci` returns the percentile confidence intervals obtained from the bootstrap
distribution. The call `score.difference()` computes the paired differences between the best
system and each of the others, and `diff.p_value(correction='holm')` returns the p-values of
those differences adjusted for multiple comparisons with Holm's method. Finally,
`score.plot()` draws the performance of the systems together with their confidence intervals.

## Citing

When CompStats is used in academic work, the first reference below, which describes the
methodology, should be cited. The second is the work it extends.

```bibtex
@article{Nava-Munoz2024,
  author  = {Nava-Mu{\~n}oz, Sergio and Graff, Mario and Escalante, Hugo Jair},
  title   = {Analysis of systems' performance in natural language processing competitions},
  journal = {Pattern Recognition Letters},
  volume  = {186},
  pages   = {346--353},
  year    = {2024},
  doi     = {10.1016/j.patrec.2024.03.010},
  url     = {https://doi.org/10.1016/j.patrec.2024.03.010}
}

@inproceedings{Nava-Munoz2023,
  author    = {Nava-Mu{\~n}oz, Sergio and Graff Guerrero, Mario and Escalante, Hugo Jair},
  title     = {Comparison of Classifiers in Challenge Scheme},
  booktitle = {Pattern Recognition},
  series    = {Lecture Notes in Computer Science},
  volume    = {13902},
  pages     = {89--98},
  year      = {2023},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-031-33783-3_9},
  url       = {https://doi.org/10.1007/978-3-031-33783-3_9}
}
```

## License

CompStats is distributed under the Apache License 2.0. See the `LICENSE` file.
