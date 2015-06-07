.. _periodic:

***************************************************
Tools for Periodic Timeseries (``gatspy.periodic``)
***************************************************

This module contains tools for analyzing periodic time series.
These tools include simple least squares spectral analysis
(e.g. :class:`~gatspy.periodic.LombScargle`) as well as non-parametric
methods (e.g. :class:`~gatspy.periodic.SuperSmoother`).

All of these fitters are classes which derive from a common
:class:`~gatspy.periodic.PeriodicModeler` class, and have a common API.

.. toctree::
   :maxdepth: 2

   lomb_scargle
   lomb_scargle_multiband
   supersmoother
   API
