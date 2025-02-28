:mod:`CompStats.metrics`
==================================

.. image:: https://github.com/INGEOTEC/CompStats/actions/workflows/test.yaml/badge.svg
		:target: https://github.com/INGEOTEC/CompStats/actions/workflows/test.yaml

.. image:: https://coveralls.io/repos/github/INGEOTEC/CompStats/badge.svg?branch=develop
		:target: https://coveralls.io/github/INGEOTEC/CompStats?branch=develop

.. image:: https://badge.fury.io/py/CompStats.svg
		:target: https://badge.fury.io/py/CompStats

.. image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/compstats-feedstock?branchName=main
	    :target: https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=20297&branchName=main

.. image:: https://img.shields.io/conda/vn/conda-forge/compstats.svg
		:target: https://anaconda.org/conda-forge/compstats

.. image:: https://img.shields.io/conda/pn/conda-forge/compstats.svg
		:target: https://anaconda.org/conda-forge/compstats

.. image:: https://readthedocs.org/projects/compstats/badge/?version=latest
		:target: https://compstats.readthedocs.io/en/latest/?badge=latest

.. image:: https://colab.research.google.com/assets/colab-badge.svg
		:target: https://colab.research.google.com/github/INGEOTEC/CompStats/blob/docs/docs/CompStats_metrics.ipynb

:py:mod:`CompStats.metrics` aims to facilitate performance measurement (with standard errors and confidence intervals) and statistical comparisons between algorithms on a single problem, wrapping the different scores and loss functions found on :py:mod:`~sklearn.metrics`.

To illustrate the use of :py:mod:`CompStats.metrics`, the following snippets show an example. The instructions load the necessary libraries, including the one to obtain the problem (e.g., digits), four different classifiers, and the last line is the score used to measure the performance and compare the algorithm. 

>>> from sklearn.svm import LinearSVC
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.datasets import load_digits
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.base import clone
>>> from CompStats.metrics import f1_score

The first step is to load the digits problem and split the dataset into training and validation sets. The second step is to estimate the parameters of a linear Support Vector Machine and predict the validation set's classes. The predictions are stored in the variable :py:attr:`hy`.

>>> X, y = load_digits(return_X_y=True)
>>> _ = train_test_split(X, y, test_size=0.3)
>>> X_train, X_val, y_train, y_val = _
>>> m = LinearSVC().fit(X_train, y_train)
>>> hy = m.predict(X_val)

Once the predictions are available, it is time to measure the algorithm's performance, as seen in the following code. It is essential to note that the API used in :py:mod:`~sklearn.metrics` is followed; the difference is that the function returns an instance with different methods that can be used to estimate different performance statistics and compare algorithms. 

>>> score = f1_score(y_val, hy, average='macro')
>>> score
<Perf(score_func=f1_score, statistic=0.9521, se=0.0097)>

The previous code shows the macro-f1 score and, in parenthesis, its standard error. The actual performance value is stored in the attributes :py:func:`~CompStats.interface.Perf.statistic` and :py:func:`~CompStats.interface.Perf.se`

>>> score.statistic, score.se
(0.9521479775366307, 0.009717884979482313)

Continuing with the example, let us assume that one wants to test another classifier on the same problem, in this case, a random forest, as can be seen in the following two lines. The second line predicts the validation set and sets it to the analysis. 

>>> ens = RandomForestClassifier().fit(X_train, y_train)
>>> score(ens.predict(X_val), name='Random Forest')
<Perf(score_func=f1_score)>
Statistic with its standard error (se)
statistic (se)
0.9720 (0.0076) <= Random Forest
0.9521 (0.0097) <= alg-1

Let us incorporate another predictions, now with Naive Bayes classifier, and Histogram Gradient Boosting as seen below.

>>> nb = GaussianNB().fit(X_train, y_train)
>>> score(nb.predict(X_val), name='Naive Bayes')
>>> hist = HistGradientBoostingClassifier().fit(X_train, y_train)
>>> score(hist.predict(X_val), name='Hist. Grad. Boost. Tree')
<Perf(score_func=f1_score)>
Statistic with its standard error (se)
statistic (se)
0.9759 (0.0068) <= Hist. Grad. Boost. Tree
0.9720 (0.0076) <= Random Forest
0.9521 (0.0097) <= alg-1
0.8266 (0.0159) <= Naive Bayes

The performance, its confidence interval (5%), and a statistical comparison (5%) between the best performing system with the rest of the algorithms is depicted in the following figure.

>>> score.plot()

.. image:: digits_perf.png

The final step is to compare the performance of the four classifiers, which can be done with the :py:func:`~CompStats.interface.Perf.difference` method, as seen next.  

>>> diff = score.difference()
>>> diff
<Difference>
difference p-values  w.r.t Hist. Grad. Boost. Tree
0.0000 <= Naive Bayes
0.0100 <= alg-1
0.3240 <= Random Forest

The class :py:class:`~CompStats.Difference` has the :py:class:`~CompStats.Difference.plot` method that can be used to depict the difference with respectto the best. 

>>> diff.plot()

.. image:: digits_difference.png

.. automodule:: CompStats.metrics
   :members: