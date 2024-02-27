.. _CompStats:

====================================
CompStats
====================================
.. image:: https://github.com/INGEOTEC/CompStats/actions/workflows/test.yaml/badge.svg
		:target: https://github.com/INGEOTEC/CompStats/actions/workflows/test.yaml

.. image:: https://coveralls.io/repos/github/INGEOTEC/CompStats/badge.svg?branch=develop
		:target: https://coveralls.io/github/INGEOTEC/CompStats?branch=develop

.. image:: https://badge.fury.io/py/CompStats.svg
		:target: https://badge.fury.io/py/CompStats

.. image:: https://readthedocs.org/projects/compstats/badge/?version=latest
		:target: https://compstats.readthedocs.io/en/latest/?badge=latest


CompStats

Quickstart Guide
====================================

The first step is to install CompStats, which is described below.

Installing CompStats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step is to install the library, which can be done using the `conda package manager <https://conda-forge.org/>`_ with the following instruction. 

.. code:: bash

	  conda install -c conda-forge CompStats

A more general approach to installing CompStats is through the use of the command pip, as illustrated in the following instruction. 

.. code:: bash

	  pip install CompStats

Libraries
^^^^^^^^^^^^^^^^^^^^^

After installing :py:class:`CompStats`, we must import the necessary libraries for our analysis. :py:class:`CompStats` relies on several Python libraries for data analysis and scientific computing.

The first line of the following code loads two functions from the :py:class:`CompStats` library. The :py:func:`~CompStats.performance.performance` function is used to calculate and analyze the performance of machine learning models. On the other hand, the :py:func:`~CompStats.performance.plot_performance` function visualizes the performance metrics calculated by :py:func:`~CompStats.performance.performance`, such as accuracy or F1 score, along with confidence intervals to help understand the variability and reliability of the performance metrics.

The second line imports two functions: :py:func:`~CompStats.performance.difference` and :py:func:`~CompStats.performance.plot_difference`; :py:func:`~CompStats.performance.difference` assesses the differences in performance between models in comparison to the best system, and :py:func:`~CompStats.performance.plot_difference` visually represents these differences relative to the best system.

The third line imports two functions: :py:func:`~CompStats.performance.all_differences` and :py:func:`~CompStats.measurements.difference_p_value`. :py:func:`~CompStats.performance.all_differences` evaluates the differences in performance between all models, and :py:func:`~CompStats.measurements.difference_p_value` estimates the p-value of the hypothesis that the difference is significantly greater than zero.

The fourth line imports the function :py:func:`multipletests` that is used for adjusting p-values when multiple hypothesis tests are performed, to control for the false discovery rate or family-wise error rate.

The rest of the lines load commonly used Python libraries.

.. code-block:: python

    >>> from CompStats import performance, plot_performance
    >>> from CompStats import difference, plot_difference
    >>> from CompStats import all_differences, difference_p_value
    >>> from statsmodels.stats.multitest import multipletests
    >>> from sklearn.metrics import f1_score
    >>> import pandas as pd

Dataset
^^^^^^^^^^^^^^^^^^^^^

Once we have set up our environment, we can explore what CompStats offers. Let's begin with a basic example of how to use CompStats for a simple statistical analysis.

To illustrate the use of CompStats, we will use a dataset included in the CompStats package. The path of the dataset is found with the following instructions. The variable :py:attr:`DATA` contains the path as shown below.  

.. code-block:: python

    >>> from CompStats.tests.test_performance import DATA
    >>> DATA
    '/usr/local/lib/python3.10/dist-packages/CompStats/tests/data.csv'


:py:attr:`DATA` contains the information to compare six systems for a multiclass classification task. The next instruction loads the data into a dataframe.

.. code-block:: python

    >>> df = pd.read_csv(DATA)

The first row of :py:attr:`df` is shown below. It can be observed that the first column contains the gold standard, identified with `y`, and the rest of the columns are the predictions performed by different systems.


Performance Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us start with the performance analysis of the different systems. The performance metric used is the weighted average F1 score. This performance is coded in the variable :py:attr:`score` as observed in the next instruction.

.. code-block:: python

    >>> score = lambda y, hy: f1_score(y, hy, average='weighted')

The next step is to compute the performance on the bootstrap samples; this is done with the function :py:func:`~CompStats.performance.performance`. The function has a few parameters; one is the :py:attr:`score`, which receives the metric used to measure the performance.

.. code-block:: python

    >>> perf = performance(df, score=score, num_samples=1000)

:py:attr:`perf` is an instance of :py:class:`~CompStats.bootstrap.StatisticSamples`, the bootstrap samples can be seen on the property :py:attr:`calls`. The first five bootstrap samples of the performance of INGEOTEC are shown below. 

.. code-block:: python

    >>> perf.calls['INGEOTEC'][:5]
    [0.37056471 0.38665852 0.36580968 0.39611708 0.39422416]

The performance of the systems, along with their confidence intervals, can be seen using the next instruction.

.. code-block:: python

    >>> face_grid = plot_performance(perf)

.. image:: performance.png


It can be observed that the best system is INGEOTEC. Although the confidence intervals provide information that helps to assess the difference in the performance of the systems, in this case, the intervals intersect. Therefore, one needs another statistical tool to determine if the difference in performance is significant.

Performance Comparison against the Winner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    
    >>> diff = difference(perf)

.. code-block:: python

    >>> diff.info
    {'best': 'INGEOTEC'}

.. code-block:: python

    >>> face_grid_diff = plot_difference(diff)

.. image:: difference.png


The difference p-value can be estimated with the following instruction.


.. code-block:: python

    >>> from CompStats.measurements import difference_p_value
    >>> p_values = difference_p_value(diff)
    >>> p_values['BoW']
    0.22
    >>> p_values['StackBoW']
    0.104


Citing
==========

If you find CompStats useful for any academic/scientific purpose, we would appreciate citations to the following reference:
  
.. code:: bibtex

    @article{Nava:2023,
    title = {{Comparison of Classifiers in Challenge Scheme}},
    year = {2023},
    journal = {Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)},
    author = {Nava-Mu{\~{n}}oz, Sergio and Graff Guerrero, Mario and Escalante, Hugo Jair},
    pages = {89--98},
    volume = {13902 LNCS},
    publisher = {Springer Science and Business Media Deutschland GmbH},
    url = {https://link.springer.com/chapter/10.1007/978-3-031-33783-3_9},
    isbn = {9783031337826},
    doi = {10.1007/978-3-031-33783-3{\_}9/COVER},
    issn = {16113349},
    keywords = {Bootstrap, Challenges, Performance}
    }


API
====================================

.. toctree::
   :maxdepth: 1

   performance_api
   measurements_api
   bootstrap_api