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


Performance
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> from CompStats import performance, plot_performance
    >>> from CompStats.tests.test_performance import DATA
    >>> from sklearn.metrics import f1_score
    >>> import pandas as pd
    >>> df = pd.read_csv(DATA)
    >>> score = lambda y, hy: f1_score(y, hy, average='weighted')
    >>> perf = performance(df, score=score)
    >>> ins = plot_performance(perf)


.. image:: performance.png


Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> from CompStats import performance, difference, plot_difference
    >>> from CompStats.tests.test_performance import DATA
    >>> from sklearn.metrics import f1_score
    >>> import pandas as pd
    >>> df = pd.read_csv(DATA)
    >>> score = lambda y, hy: f1_score(y, hy, average='weighted')
    >>> perf = performance(df, score=score)
    >>> diff = difference(perf)
    >>> ins = plot_difference(diff)    

.. image:: difference.png


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

