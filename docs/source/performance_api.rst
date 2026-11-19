:mod:`CompStats.performance`
==================================

:mod:`CompStats.performance` is the earlier, functional API: it operates on a
:py:class:`pandas.DataFrame` with one gold column plus one column per system. It is kept for the
multi-metric competition reports of the paper (coefficient of variation, PPI and distance to the
best). New code should use :py:class:`~CompStats.interface.Perf` and
:py:class:`~CompStats.interface.Difference`.

.. automodule:: CompStats.performance
   :members:
