from __future__ import division, print_function

import numpy as np

from .optimizer import LinearScanOptimizer


class PeriodicModeler(object):
    """Base class for periodic modeling"""
    def __init__(self, optimizer=None, *args, **kwargs):
        if optimizer is None:
            optimizer = LinearScanOptimizer()
        if not hasattr(optimizer, 'best_period'):
            raise ValueError("optimizer must be an PeriodicOptimizer instance. "
                             "{0} not valid".format(optimizer))
        self.optimizer = optimizer
        self.args = args
        self.kwargs = kwargs
        self._best_period = None

    def fit(self, t, y, dy=None, filts=None):
        """Fit the multiterm Periodogram model to the data.

        Parameters
        ----------
        t : array_like, one-dimensional
            sequence of observation times
        y : array_like, one-dimensional
            sequence of observed values
        dy : float or array_like (optional)
            errors on observed values
        filts : array_like (optional)
            The array specifying the filter/bandpass for each observation.
        """
        if filts is None:
            self.t, self.y, self.dy = np.broadcast_arrays(t, y, dy)
            self.filts = None
        else:
            self.t, self.y, self.dy, self.filts = np.broadcast_arrays(t, y, dy,
                                                                      filts)
        self._fit(self.t, self.y, self.dy, filts=self.filts)
        return self

    def predict(self, t, filts=None, period=None):
        """Compute the best-fit model at ``t`` for a given frequency omega

        Parameters
        ----------
        t : float or array_like
            times at which to predict
        filts : array_like (optional)
            the array specifying the filter/bandpass for each observation. This
            is used only in multiband periodograms.
        period : float (optional)
            The period at which to compute the model. If not specified, it
            will be computed via the optimizer provided at initialization.

        Returns
        -------
        y : np.ndarray
            predicted model values at times t
        """
        if period is None:
            period = self.best_period

        if filts is None:
            t = np.asarray(t)
            if self.filts is not None:
                raise ValueError("filts must be passed")
            result = self._predict(t.ravel(), filts=filts, period=period)
        else:
            t, filts = np.broadcast_arrays(t, filts)
            if self.filts is None:
                raise ValueError("filts passed to predict(), but not to fit()")
            result = self._predict(t.ravel(), filts=filts.ravel(),
                                   period=period)
        return result.reshape(t.shape)

    def score(self, periods):
        """Compute the score for the given period or periods

        Parameters
        ----------
        periods : float or array_like
            Array of angular frequencies at which to compute
            the periodogram

        Returns
        -------
        scores : np.ndarray
            Array of normalized powers (between 0 and 1) for each frequency.
            Shape of scores matches the shape of the provided periods.
        """
        periods = np.asarray(periods)
        return self._score(periods.ravel()).reshape(periods.shape)

    periodogram = score

    @property
    def best_period(self):
        """Lazy evaluation of the best period given the model"""
        if not hasattr(self, '_best_period') or self._best_period is None:
            self._best_period = self._calc_best_period()
        return self._best_period

    def find_best_periods(self, n_periods=5, return_scores=False):
        """Find the top several best periods for the model"""
        return self.optimizer.find_best_periods(self, n_periods,
                                                return_scores=return_scores)

    def _calc_best_period(self):
        """Compute the best period using the optimizer"""
        return self.optimizer.best_period(self)

    def _score(self, periods):
        """Compute the score of the model given the periods"""
        raise NotImplementedError()

    def _fit(self, t, y, dy, filts):
        """Fit the model to the given data"""
        raise NotImplementedError()

    def _predict(self, t, filts, period):
        """Predict the model values at the given times & filters"""
        raise NotImplementedError()
        
