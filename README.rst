====================================
CompStats
====================================
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

Collaborative competitions have gained popularity in the scientific and technological fields. These competitions involve defining tasks, selecting evaluation scores, and devising result verification methods. In the standard scenario, participants receive a training set and are expected to provide a solution for a held-out dataset kept by organizers. An essential challenge for organizers arises when comparing algorithms' performance, assessing multiple participants, and ranking them. Statistical tools are often used for this purpose; however, traditional statistical methods often fail to capture decisive differences between systems' performance. CompStats implements an evaluation methodology for statistically analyzing competition results and competition. CompStats offers several advantages, including off-the-shell comparisons with correction mechanisms and the inclusion of confidence intervals. 

To illustrate the use of `CompStats`, the following snippets show an example. The instructions load the necessary libraries, including the one to obtain the problem (e.g., digits), three different classifiers, and the last line is the score used to measure the performance and compare the algorithm. 

>>> from sklearn.svm import LinearSVC
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.datasets import load_digits
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.base import clone
>>> from CompStats.metrics import f1_score

The first step is to load the digits problem and split the dataset into training and validation sets. The second step is to estimate the parameters of a linear Support Vector Machine and predict the validation set's classes. The predictions are stored in the variable `hy`.

>>> X, y = load_digits(return_X_y=True)
>>> _ = train_test_split(X, y, test_size=0.3)
>>> X_train, X_val, y_train, y_val = _
>>> m = LinearSVC().fit(X_train, y_train)
>>> hy = m.predict(X_val)

Once the predictions are available, it is time to measure the algorithm's performance, as seen in the following code. It is essential to note that the API used in `sklearn.metrics` is followed; the difference is that the function returns an instance with different methods that can be used to estimate different performance statistics and compare algorithms. 

>>> score = f1_score(y_val, hy, average='macro')
>>> score
<Perf(score_func=f1_score)>
Statistic with its standard error (se)
statistic (se)
0.9332 (0.0113) <= alg-1

The previous code shows the macro-f1 score and, in parenthesis, its standard error. The actual performance value is stored in the `statistic` function.

>>> score.statistic
{'alg-1': 0.9332035615949114}

Continuing with the example, let us assume that one wants to test another classifier on the same problem, in this case, a random forest, as can be seen in the following two lines. The second line predicts the validation set and sets it to the analysis. 

>>> ens = RandomForestClassifier().fit(X_train, y_train)
>>> score(ens.predict(X_val), name='Random Forest')
<Perf(score_func=f1_score)>
Statistic with its standard error (se)
statistic (se)
0.9756 (0.0061) <= Random Forest
0.9332 (0.0113) <= alg-1

Let us incorporate another prediction, now with the Naive Bayes classifier, as seen below.

>>> nb = GaussianNB().fit(X_train, y_train)
>>> score(nb.predict(X_val), name='Naive Bayes')
<Perf(score_func=f1_score)>
Statistic with its standard error (se)
statistic (se)
0.9756 (0.0061) <= Random Forest
0.9332 (0.0113) <= alg-1
0.8198 (0.0144) <= Naive Bayes

The final step is to compare the performance of the three classifiers, which can be done with the `difference` method, as seen next.  

>>> diff = score.difference()
>>> diff
<Difference>
difference p-values w.r.t Random Forest
0.0000 <= alg-1
0.0000 <= Naive Bayes

The class `Difference` has the `plot` method that can be used to depict the difference with respect to the best. 

>>> diff.plot()

.. image:: https://github.com/INGEOTEC/CompStats/raw/docs/docs/source/digits_difference.png
