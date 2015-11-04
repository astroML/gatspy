from __future__ import division, print_function, absolute_import

__all__ = ['LombScargle', 'LombScargleAstroML']

import warnings

import numpy as np

from .modeler import PeriodicModeler
from ._least_squares_mixin import LeastSquaresMixin


class LombScargle(LeastSquaresMixin, PeriodicModeler):
    """Lomb-Scargle Periodogram Implementation

    This is a generalized periodogram implementation using the matrix formalism
    outlined in VanderPlas & Ivezic 2015. It allows computation of periodograms
    and best-fit models for both the classic normalized periodogram and
    truncated Fourier series generalizations.

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
    center_data : boolean (default = True)
        If True, then compute the weighted mean of the input data and subtract
        before fitting the model.
    fit_offset : boolean (default = True)
        If True, then fit a floating-mean sinusoid model.
    Nterms : int (default = 1)
        Number of Fourier frequencies to fit in the model
    regularization : float, vector or None (default = None)
        If specified, then add this regularization penalty to the
        least squares fit.
    regularize_by_trace : boolean (default = True)
        If True, multiply regularization by the trace of the matrix
    fit_period : bool (optional)
        If True, then fit for the best period when fit() method is called.
    optimizer_kwds : dict (optional)
        Dictionary of keyword arguments for constructing the optimizer. For
        example, silence optimizer output with `optimizer_kwds={"quiet": True}`.

    Examples
    --------
    >>> rng = np.random.RandomState(0)
    >>> t = 100 * rng.rand(100)
    >>> dy = 0.1
    >>> omega = 10
    >>> y = np.sin(omega * t) + dy * rng.randn(100)
    >>> ls = LombScargle().fit(t, y, dy)
    >>> ls.optimizer.period_range = (0.2, 1.2)
    >>> ls.best_period
    Finding optimal frequency:
     - Estimated peak width = 0.0639
     - Using 5 steps per peak; omega_step = 0.0128
     - User-specified period range:  0.2 to 1.2
     - Computing periods at 2051 steps
    Zooming-in on 5 candidate peaks:
     - Computing periods at 1000 steps
    0.62827068275990694
    >>> ls.predict([0, 0.5])
    array([-0.01445853, -0.92762251])

    See Also
    --------
    LombScargleAstroML
    LombScargleMultiband
    LombScargleMultibandFast
    """
    def __init__(self, optimizer=None, center_data=True, fit_offset=True,
                 Nterms=1, regularization=None, regularize_by_trace=True,
                 fit_period=False, optimizer_kwds=None):
        self.center_data = center_data
        self.fit_offset = fit_offset
        self.Nterms = int(Nterms)
        self.regularization = regularization
        self.regularize_by_trace = regularize_by_trace
        PeriodicModeler.__init__(self, optimizer, fit_period=fit_period,
                                 optimizer_kwds=optimizer_kwds)

        if not self.center_data and not self.fit_offset:
            warnings.warn("Not centering data or fitting offset can lead "
                          "to poor results")

        if self.Nterms < 0:
            raise ValueError("Nterms must be non-negative")

        if self.Nterms == 0 and not fit_offset:
            raise ValueError("Empty model: try larger Nterms.")

    def _construct_X(self, omega, weighted=True, **kwargs):
        """Construct the design matrix for the problem"""
        t = kwargs.get('t', self.t)
        dy = kwargs.get('dy', self.dy)
        fit_offset = kwargs.get('fit_offset', self.fit_offset)

        if fit_offset:
            offsets = [np.ones(len(t))]
        else:
            offsets = []

        cols = sum(([np.sin((i + 1) * omega * t),
                     np.cos((i + 1) * omega * t)]
                    for i in range(self.Nterms)), offsets)

        if weighted:
            return np.transpose(np.vstack(cols) / dy)
        else:
            return np.transpose(np.vstack(cols))

    def _fit(self, t, y, dy):
        self.yw_ = self._construct_y(weighted=True)
        self.ymean_ = self._compute_ymean()

    def _predict(self, t, period):
        omega = 2 * np.pi / period
        theta = self._best_params(omega)
        X = self._construct_X(omega, weighted=False, t=t)
        if self.center_data:
            return self.ymean_ + np.dot(X, theta)
        else:
            return np.dot(X, theta)

    def _score(self, periods):
        return LeastSquaresMixin._score(self, periods)


class LombScargleAstroML(LombScargle):
    """Lomb-Scargle Periodogram Implementation using AstroML

    This is a generalized periodogram implementation which uses the periodogram
    functions from astroML. Compared to LombScargle, this implementation is
    both faster and more memory-efficient.

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
    Nterms : int (default = 1)
        Number of terms for the fit. Only Nterms=1 is currently supported.
    center_data : boolean (default = True)
        If True, then compute the weighted mean of the input data and subtract
        before fitting the model.
    fit_offset : boolean (default = True)
        If True, then fit a floating-mean sinusoid model.
    slow_version : boolean (default = False)
        If True, use the slower pure-python version from astroML. Otherwise,
        use the faster version of the code from astroML_addons
    fit_period : bool (optional)
        If True, then fit for the best period when fit() method is called.
    optimizer_kwds : dict (optional)
        Dictionary of keyword arguments for constructing the optimizer. For
        example, silence optimizer output with `optimizer_kwds={"quiet": True}`.

    Examples
    --------
    >>> rng = np.random.RandomState(0)
    >>> t = 100 * rng.rand(100)
    >>> dy = 0.1
    >>> omega = 10
    >>> y = np.sin(omega * t) + dy * rng.randn(100)
    >>> ls = LombScargleAstroML().fit(t, y, dy)
    >>> ls.optimizer.period_range = (0.2, 1.2)
    >>> ls.best_period
    Finding optimal frequency:
     - Estimated peak width = 0.0639
     - Using 5 steps per peak; omega_step = 0.0128
     - User-specified period range:  0.2 to 1.2
     - Computing periods at 2051 steps
    Zooming-in on 5 candidate peaks:
     - Computing periods at 1000 steps
    0.62827068275990694

    See Also
    --------
    LombScargle
    LombScargleMultiband
    LombScargleMultibandFast
    """
    def __init__(self, optimizer=None, Nterms=1, fit_offset=True,
                 center_data=True, slow_version=False,
                 fit_period=False, optimizer_kwds=None):
        if Nterms != 1:
            raise ValueError("Only Nterms=1 is supported")

        LombScargle.__init__(self, optimizer=optimizer, Nterms=1,
                             center_data=center_data, fit_offset=fit_offset,
                             fit_period=fit_period,
                             optimizer_kwds=optimizer_kwds)
        if slow_version:
            from astroML.time_series._periodogram import lomb_scargle
        else:
            from astroML.time_series import lomb_scargle
        self._LS_func = lomb_scargle

    def _score(self, periods):
        return self._LS_func(self.t, self.y, self.dy, 2 * np.pi / periods,
                             generalized=self.fit_offset,
                             subtract_mean=self.center_data)
