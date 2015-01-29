from __future__ import division, print_function, absolute_import

import numpy as np

from .modeler import PeriodicModeler


class LeastSquaresMixin(object):
    """Mixin for matrix-based Least Squares periodic analysis"""
    def _construct_X(self, omega, weighted=True, **kwargs):
        raise NotImplementedError()

    def _construct_y(self, weighted=True, **kwargs):
        raise NotImplementedError()

    def _construct_X_M(self, omega, **kwargs):
        """Construct the weighted normal matrix of the problem"""
        X = self._construct_X(omega, weighted=True, **kwargs)
        M = np.dot(X.T, X)

        if hasattr(self, 'regularization') and self.regularization is not None:
            diag = M.ravel(order='K')[::M.shape[0] + 1]
            if self.regularize_by_trace:
                diag += diag.sum() * np.asarray(self.regularization)
            else:
                diag += np.asarray(self.regularization)

        return X, M

    def _compute_ymean(self, **kwargs):
        """Compute the (weighted) mean of the y data"""
        y = np.asarray(kwargs.get('y', self.y))
        dy = np.asarray(kwargs.get('dy', self.dy))

        # if dy is a scalar, we use the simple mean
        if dy.size == 1:
            return np.mean(y)
        else:
            w = 1 / dy ** 2
            return np.dot(y, w) / w.sum()

    def _construct_y(self, weighted=True, **kwargs):
        y = kwargs.get('y', self.y)
        dy = kwargs.get('dy', self.dy)
        center_data = kwargs.get('center_data', self.center_data)

        y = np.asarray(y)
        dy = np.asarray(dy)

        if center_data:
            y = y - self._compute_ymean(y=y, dy=dy)

        if weighted:
            return y / dy
        else:
            return y

    def _best_params(self, omega):
        Xw, XTX = self._construct_X_M(omega)
        XTy = np.dot(Xw.T, self.yw_)
        return np.linalg.solve(XTX, XTy)

    def _score(self, periods):
        omegas = 2 * np.pi / periods

        # Set up the reference chi2. Note that this entire function would
        # be much simpler if we did not allow center_data=False.
        # We keep it just to make sure our math is correct
        chi2_0 = np.dot(self.yw_.T, self.yw_)
        if self.center_data:
            chi2_ref = chi2_0
        else:
            yref = self._construct_y(weighted=True, center_data=True)
            chi2_ref = np.dot(yref.T, yref)
        chi2_0_minus_chi2 = np.zeros(omegas.size, dtype=float)

        # Iterate through the omegas and compute the power for each
        for i, omega in enumerate(omegas.flat):
            Xw, XTX = self._construct_X_M(omega)
            XTy = np.dot(Xw.T, self.yw_)
            chi2_0_minus_chi2[i] = np.dot(XTy.T, np.linalg.solve(XTX, XTy))

        # construct and return the power from the chi2 difference
        if self.center_data:
            P = chi2_0_minus_chi2 / chi2_ref
        else:
            P = 1 + (chi2_0_minus_chi2 - chi2_0) / chi2_ref

        return P


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

    Examples
    --------
    >>> rng = np.random.RandomState(0)
    >>> t = 100 * rng.rand(100)
    >>> dy = 0.1
    >>> omega = 10
    >>> y = np.sin(omega * t) + dy * rng.randn(100)
    >>> ls = LombScargle().fit(t, y, dy)
    >>> ls.best_period
    0.62827393156409295
    >>> ls.score(ls.best_period)
    array(0.9815178000850804)
    >>> ls.predict([0, 0.5])
    array([-0.01213809, -0.92700951])

    See Also
    --------
    LombScargleAstroML
    LombScargleMultiband
    LombScargleMultibandFast
    """
    def __init__(self, optimizer=None, center_data=True, fit_offset=True,
                 Nterms=1, regularization=None, regularize_by_trace=True):
        self.center_data = center_data
        self.fit_offset = fit_offset
        self.Nterms = int(Nterms)
        self.regularization = regularization
        self.regularize_by_trace = regularize_by_trace
        PeriodicModeler.__init__(self, optimizer)

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
        return self

    def _predict(self, t, period):
        omega = 2 * np.pi / period
        t = np.asarray(t)
        outshape = t.shape
        theta = self._best_params(omega)
        X = self._construct_X(omega, weighted=False, t=t.ravel())
        return np.reshape(self.ymean_ + np.dot(X, theta), outshape)

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

    Examples
    --------
    >>> rng = np.random.RandomState(0)
    >>> t = 100 * rng.rand(100)
    >>> dy = 0.1
    >>> omega = 10
    >>> y = np.sin(omega * t) + dy * rng.randn(100)
    >>> ls = LombScargleAstroML().fit(t, y, dy)
    >>> ls.best_period
    0.62827393156409295

    See Also
    --------
    LombScargle
    LombScargleMultiband
    LombScargleMultibandFast
    """
    def __init__(self, optimizer=None, Nterms=1, fit_offset=True,
                 center_data=True, slow_version=False):
        if Nterms != 1:
            raise ValueError("Only Nterms=1 is supported")

        LombScargle.__init__(self, optimizer=optimizer, Nterms=1,
                             center_data=center_data, fit_offset=fit_offset)
        if slow_version:
            from astroML.time_series._periodogram import lomb_scargle
        else:
            from astroML.time_series import lomb_scargle
        self._LS_func = lomb_scargle

    def _score(self, periods):
        return self._LS_func(self.t, self.y, self.dy, 2 * np.pi / periods,
                             generalized=self.fit_offset,
                             subtract_mean=self.center_data)
