"""
Naive Multiband Methods

This basically amounts to a band-by-band single band approach, followed by
some sort of majority vote among the peaks of the individual periodograms.
"""
from __future__ import division, print_function, absolute_import

__all__ = ['NaiveMultiband']

import numpy as np
from scipy.stats import mode

from .modeler import PeriodicModelerMultiband
from .lomb_scargle import LombScargle


def mode_in_range(a, axis=0, tol=1E-3):
    """Find the mode of values to within a certain range"""
    a_trunc = a // tol
    vals, counts = mode(a_trunc, axis)
    mask = (a_trunc == vals)
    # mean of each row
    return np.sum(a * mask, axis) / np.sum(mask, axis)


class NaiveMultiband(PeriodicModelerMultiband):
    """Naive version of multiband fitting

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
    BaseModel : PeriodicModeler instance
        Single-band model to use on data from each band.
    fit_period : bool (optional)
        If True, then fit for the best period when fit() method is called.
    optimizer_kwds : dict (optional)
        Dictionary of keyword arguments for constructing the optimizer. For
        example, silence optimizer output with `optimizer_kwds={"quiet": True}`.
    *args, **kwargs :
        additional arguments are passed to BaseModel on construction.
    """
    def __init__(self, optimizer=None, BaseModel=LombScargle,
                 fit_period=False, optimizer_kwds=None,
                 *args, **kwargs):
        self.BaseModel = BaseModel
        self.args = args
        self.kwargs = kwargs
        PeriodicModelerMultiband.__init__(self, optimizer,
                                          fit_period=fit_period,
                                          optimizer_kwds=optimizer_kwds)

    def _fit(self, t, y, dy, filts):
        t, y, dy, filts = np.broadcast_arrays(t, y, dy, filts)
        unique_filts = np.unique(filts)

        masks = [(filts == filt) for filt in unique_filts]
        self.models_ = dict([(filt,
                              self.BaseModel(self.optimizer,
                                             *self.args,
                                             **self.kwargs).fit(t[mask],
                                                                y[mask],
                                                                dy[mask]))
                             for filt, mask in zip(unique_filts, masks)])

    def _predict(self, t, filts, period):
        result = np.zeros_like(t)
        for filt, model in self.models_.items():
            mask = (filts == filt)
            result[mask] = model.predict(t[mask], period=period)
        return result

    def _score(self, periods):
        raise NotImplementedError("score is not implmented for NaiveMultiband")

    def scores(self, periods):
        """Compute the scores under the various models

        Parameters
        ----------
        periods : array_like
            array of periods at which to compute scores

        Returns
        -------
        scores : dict
            Dictionary of scores. Dictionary keys are the unique filter names
            passed to fit()
        """
        return dict([(filt, model.score(periods))
                     for (filt, model) in self.models_.items()])

    def best_periods(self):
        """Compute the scores under the various models

        Parameters
        ----------
        periods : array_like
            array of periods at which to compute scores

        Returns
        -------
        best_periods : dict
            Dictionary of best periods. Dictionary keys are the unique filter
            names passed to fit()
        """
        for (key, model) in self.models_.items():
            model.optimizer = self.optimizer

        return dict((filt, model.best_period)
                    for (filt, model) in self.models_.items())

    @property
    def best_period(self):
        best_periods = np.asarray(list(self.best_periods().values()))
        return mode_in_range(best_periods, tol=1E-2)
