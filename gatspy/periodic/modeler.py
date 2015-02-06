from __future__ import division, print_function

import numpy as np

from .optimizer import LinearScanOptimizer


class PeriodicModeler(object):
    """Base class for periodic modeling"""
    def __init__(self, optimizer=None, *args, **kwargs):
        if optimizer is None:
            optimizer = LinearScanOptimizer()
        if not hasattr(optimizer, 'best_period'):
            raise ValueError("optimizer must be a PeriodicOptimizer instance: "
                             "{0} has no best_period method".format(optimizer))
        self.optimizer = optimizer
        self.args = args
        self.kwargs = kwargs
        self._best_period = None

    def fit(self, t, y, dy=None):
        """Fit the multiterm Periodogram model to the data.

        Parameters
        ----------
        t : array_like, one-dimensional
            sequence of observation times
        y : array_like, one-dimensional
            sequence of observed values
        dy : float or array_like (optional)
            errors on observed values
        """
        self.t, self.y, self.dy = np.broadcast_arrays(t, y, dy)
        self._fit(self.t, self.y, self.dy)
        self._best_period = None  # reset best period in case of refitting
        return self

    def predict(self, t, period=None):
        """Compute the best-fit model at ``t`` for a given frequency omega

        Parameters
        ----------
        t : float or array_like
            times at which to predict
        period : float (optional)
            The period at which to compute the model. If not specified, it
            will be computed via the optimizer provided at initialization.

        Returns
        -------
        y : np.ndarray
            predicted model values at times t
        """
        t = np.asarray(t)
        if period is None:
            period = self.best_period
        result = self._predict(t.ravel(), period=period)
        return result.reshape(t.shape)

    def score_frequency_grid(self, f0, df, N):
        """Compute the score on a frequency grid.

        Some models can compute results faster if the inputs are passed in this
        manner.

        Parameters
        ----------
        f0, df, N : (float, float, int)
            parameters describing the frequency grid freq = f0 + df * arange(N)
            Note that these are frequencies, not angular frequencies.

        Returns
        -------
        score : ndarray
            the length-N array giving the score at each frequency
        """
        return self._score_frequency_grid(f0, df, N)

    def score(self, periods):
        """Compute the periodogram for the given period or periods

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
        if self._best_period is None:
            self._best_period = self._calc_best_period()
        return self._best_period

    def find_best_periods(self, n_periods=5, return_scores=False):
        """Find the top several best periods for the model"""
        return self.optimizer.find_best_periods(self, n_periods,
                                                return_scores=return_scores)

    def _calc_best_period(self):
        """Compute the best period using the optimizer"""
        return self.optimizer.best_period(self)

    # The following methods should be overloaded by derived classes:

    def _score_frequency_grid(self, f0, df, N):
        freq = f0 + df * np.arange(N)
        return self._score(1. / freq)

    def _score(self, periods):
        """Compute the score of the model given the periods"""
        raise NotImplementedError()

    def _fit(self, t, y, dy):
        """Fit the model to the given data"""
        raise NotImplementedError()

    def _predict(self, t, period):
        """Predict the model values at the given times"""
        raise NotImplementedError()


class PeriodicModelerMultiband(PeriodicModeler):
    """Base class for periodic modeling on multiband data"""

    def fit(self, t, y, dy=None, filts=0):
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
        self.unique_filts_ = np.unique(filts)
        self.t, self.y, self.dy, self.filts = np.broadcast_arrays(t, y, dy,
                                                                  filts)
        self._fit(self.t, self.y, self.dy, self.filts)
        self._best_period = None  # reset best period in case of refitting
        return self

    def predict(self, t, filts, period=None):
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
        unique_filts = set(np.unique(filts))
        if not unique_filts.issubset(self.unique_filts_):
            raise ValueError("filts does not match training data: "
                             "input: {0} output: {1}"
                             "".format(set(self.unique_filts_),
                                       set(unique_filts)))
        t, filts = np.broadcast_arrays(t, filts)

        if period is None:
            period = self.best_period

        result = self._predict(t.ravel(), filts=filts.ravel(), period=period)
        return result.reshape(t.shape)

    # The following methods should be overloaded by derived classes:

    def _score(self, periods):
        """Compute the score of the model given the periods"""
        raise NotImplementedError()

    def _fit(self, t, y, dy, filts):
        """Fit the model to the given data"""
        raise NotImplementedError()

    def _predict(self, t, filts, period):
        """Predict the model values at the given times & filters"""
        raise NotImplementedError()
