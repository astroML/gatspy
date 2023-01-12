from __future__ import division, print_function, absolute_import

__all__ = ['TrendedLombScargleMultiband', 'PolyTrend_LombScargleMultiband']

import warnings

import numpy as np

from .modeler import PeriodicModeler
from .lomb_scargle import LombScargle
from .lomb_scargle_multiband import LombScargleMultiband


class TrendedLombScargleMultiband(LombScargleMultiband):
    """Trended Multiband Lomb-Scargle Periodogram Implementation

    This is a generalized multiband periodogram implementation using the matrix
    formalism outlined in VanderPlas & Ivezic 2015. It fits both a floating mean
    and a trend parameter in each band.

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
    Nterms_base : integer (default = 1)
        number of frequency terms to use for the base model common to all bands
    Nterms_band : integer (default = 1)
        number of frequency terms to use for the residuals between the base
        model and each individual band
    reg_base : float or None (default = None)
        amount of regularization to use on the base model parameters
    reg_band : float or None (default = 1E-6)
        amount of regularization to use on the band model parameters
    regularize_by_trace : bool (default = True)
        if True, then regularization is expressed in units of the trace of
        the normal matrix
    center_data : boolean (default = True)
        if True, then center the y data prior to the fit
    optimizer_kwds : dict (optional)
        Dictionary of keyword arguments for constructing the optimizer. For
        example, silence optimizer output with `optimizer_kwds={"quiet": True}`.

    See Also
    --------
    LombScargle
    LombScargleMultiband
    TrendedLombScargle
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
        #  [len(t), (2 + 2 * Nterms_base + nfilts * (2 + 2 * Nterms_band))]

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


class PolyTrend_LombScargleMultiband(LombScargleMultiband):
    """Trended Multiband Lomb-Scargle Periodogram Implementation

    This is a generalized multiband periodogram implementation using the matrix
    formalism outlined in VanderPlas & Ivezic 2015. It fits both a floating mean
    and a trend parameter in each band.

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
    Nterms_base : integer (default = 1)
        number of frequency terms to use for the base model common to all bands
    Nterms_band : integer (default = 1)
        number of frequency terms to use for the residuals between the base
        model and each individual band
    reg_base : float or None (default = None)
        amount of regularization to use on the base model parameters
    reg_band : float or None (default = 1E-6)
        amount of regularization to use on the band model parameters
    regularize_by_trace : bool (default = True)
        if True, then regularization is expressed in units of the trace of
        the normal matrix
    center_data : boolean (default = True)
        if True, then center the y data prior to the fit
    optimizer_kwds : dict (optional)
        Dictionary of keyword arguments for constructing the optimizer. For
        example, silence optimizer output with `optimizer_kwds={"quiet": True}`.

    See Also
    --------
    LombScargle
    LombScargleMultiband
    TrendedLombScargle
    """
    
    def __init__(self, optimizer=None,
                 poly_trend_order=2,
                 Nterms_base=1, Nterms_band=1,
                 reg_base=None, reg_band=1E-6, regularize_by_trace=True,
                 center_data=True, fit_period=False, optimizer_kwds=None):
        self.poly_trend_order = poly_trend_order
        
        LombScargleMultiband.__init__(
            self, optimizer=optimizer,
            Nterms_base=Nterms_base, Nterms_band=Nterms_band,
            reg_base=reg_base, reg_band=reg_band,
            regularize_by_trace=regularize_by_trace,
            center_data=center_data, fit_period=fit_period,
            optimizer_kwds=optimizer_kwds,
        )
    
    def _construct_regularization(self):
        if self.reg_base is None and self.reg_band is None:
            reg = 0
        else:
            Nbase = (self.poly_trend_order + 1) + 2 * self.Nterms_base
            Nband = (self.poly_trend_order + 1) + 2 * self.Nterms_band
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
        #  [len(t), ((poly_trend_order + 1) + 2 * Nterms_base +
        #            nfilts * ((poly_trend_order + 1) + 2 * Nterms_band))]
        
        # TODO: the building of the matrix could be more efficient
        cols = [np.ones(len(t))]    # Coefficients for constant param
        
        for cur_poly_trend_order in range(1, self.poly_trend_order + 1):
            # Coefficients for current polynomial trend order
            cols.append(np.copy(t**cur_poly_trend_order))
        
        cols = sum(([np.sin((i + 1) * omega * t),
                     np.cos((i + 1) * omega * t)]
                    for i in range(self.Nterms_base)), cols)
        
        for filt in self.unique_filts_:
            cols.append(np.ones(len(t)))
            
            for cur_poly_trend_order in range(1, self.poly_trend_order + 1):
                # Coefficients for current polynomial trend order
                cols.append(np.copy(t**cur_poly_trend_order))
            
            cols = sum(([np.sin((i + 1) * omega * t),
                         np.cos((i + 1) * omega * t)]
                        for i in range(self.Nterms_band)), cols)
            mask = (filts == filt)
            for i in range((-1 * (self.poly_trend_order + 1)) -
                           (2 * self.Nterms_band), 0):
                cols[i][~mask] = 0
        
        if weighted:
            return np.transpose(np.vstack(cols) / dy)
        else:
            return np.transpose(np.vstack(cols))
