from __future__ import division, print_function, absolute_import

__all__ = ['TrendedLombScargleMultiband']

import warnings

import numpy as np

from .modeler import PeriodicModeler
from .lomb_scargle import LombScargle
from .lomb_scargle_multiband import LombScargleMultiband


class TrendedLombScargleMultiband(LombScargleMultiband):
    """Trended Lomb-Scargle Periodogram Implementation

    This is a generalized periodogram implementation using the matrix formalism
    outlined in VanderPlas & Ivezic 2015. It fits both a floating mean
    and a trend parameter (as opposed to the `LombScargle` class, which 
    fits only the mean).

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
    >>> slope = 2.
    >>> y = np.sin(omega * t) + slope * t + dy * rng.randn(100)
    >>> ls = TrendedLombScargle().fit(t, y, dy)
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
    array([-0.01144474,  0.07567192])

    See Also
    --------
    LombScargle
    LombScargleAstroML
    LombScargleMultiband
    LombScargleMultibandFast
    """
    
    def _construct_regularization(self):
        if self.reg_base is None and self.reg_band is None:
            reg = 0
        else:
            Nbase = 2 + 2 * self.Nterms_base
            Nband = 2 + 2 * self.Nterms_band
            reg = np.zeros(Nbase + len(self.unique_filts_) * Nband)
            if self.reg_base is not None:
                reg[:Nbase] = self.reg_base
            if self.reg_band is not None:
                reg[Nbase:] = self.reg_band
        return reg
    
    def _construct_X(self, omega, weighted=True, **kwargs):
        t = kwargs.get('t', self.t)
        dy = kwargs.get('dy', self.dy)
        filts = kwargs.get('filts', self.filts)

        # X is a huge-ass matrix that has lots of zeros depending on
        # which filters are present...
        #
        # huge-ass, quantitatively speaking, is of shape
        #  [len(t), (1 + 1 + 2 * Nterms_base + nfilts * (1 + 1 + 2 * Nterms_band))]

        # TODO: the building of the matrix could be more efficient
        cols = [np.ones(len(t))]
        cols.append(np.copy(t))   #  coefficients for trend parameter       
        cols = sum(([np.sin((i + 1) * omega * t),
                     np.cos((i + 1) * omega * t)]
                    for i in range(self.Nterms_base)), cols)
        
        for filt in self.unique_filts_:
            cols.append(np.ones(len(t)))
            cols.append(np.copy(t))   #  coefficients for trend parameter (in filt)
            cols = sum(([np.sin((i + 1) * omega * t),
                         np.cos((i + 1) * omega * t)]
                        for i in range(self.Nterms_band)), cols)
            mask = (filts == filt)
            for i in range(-2 - (2 * self.Nterms_band), 0):
                cols[i][~mask] = 0
        
        if weighted:
            return np.transpose(np.vstack(cols) / dy)
        else:
            return np.transpose(np.vstack(cols))
