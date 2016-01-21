"""
Supersmoother code for periodic modeling
"""
from __future__ import print_function, division, absolute_import

__all__ = ['SuperSmoother', 'SuperSmootherMultiband']

import numpy as np

try:
    import supersmoother as ssm
except ImportError:
    ssm = None

from .modeler import PeriodicModeler, PeriodicModelerMultiband


class SuperSmoother(PeriodicModeler):
    """Periodogram based on Friedman's SuperSmoother.

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
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
    >>> ssm = SuperSmoother().fit(t, y, dy)
    >>> ssm.optimizer.period_range = (0.2, 1.2)
    >>> ssm.best_period
    Finding optimal frequency:
     - Estimated peak width = 0.0639
     - Using 5 steps per peak; omega_step = 0.0128
     - User-specified period range:  0.2 to 1.2
     - Computing periods at 2051 steps
    Zooming-in on 5 candidate peaks:
     - Computing periods at 1000 steps
    0.62819846183431927
    >>> ssm.predict([0, 0.5])
    array([-0.02195035, -0.92119149])

    See Also
    --------
    LombScargle
    """
    def __init__(self, optimizer=None,
                 fit_period=False, optimizer_kwds=None):
        if ssm is None:
            raise ImportError("Package supersmoother is required. "
                              "Use ``pip install supersmoother`` to install")
        PeriodicModeler.__init__(self, optimizer,
                                 fit_period=fit_period,
                                 optimizer_kwds=optimizer_kwds)

    def _fit(self, t, y, dy):
        # TODO: this should actually be a weighted median, probably...
        if dy.size == 1:
            mu = np.mean(y)
        else:
            mu = np.average(y, weights=1 / dy ** 2)
        self.baseline_err = np.mean(abs((y - mu) / dy))

    def _predict(self, t, period):
        model = ssm.SuperSmoother(period=period).fit(self.t, self.y, self.dy)
        return model.predict(t)

    def _score(self, periods):
        return np.asarray([1 - (ssm.SuperSmoother(period=p)
                                .fit(self.t, self.y, self.dy)
                                .cv_error(skip_endpoints=False)
                                / self.baseline_err)
                           for p in periods])


class SuperSmootherMultiband(PeriodicModelerMultiband):
    """
    Simple multi-band SuperSmoother, with each band smoothed independently

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
    BaseModel : class type (default = SuperSmoother)
        The base model to use for each individual band.
    fit_period : bool (optional)
        If True, then fit for the best period when fit() method is called.
    optimizer_kwds : dict (optional)
        Dictionary of keyword arguments for constructing the optimizer. For
        example, silence optimizer output with `optimizer_kwds={"quiet": True}`.
    """
    def __init__(self, optimizer=None, BaseModel=SuperSmoother,
                 fit_period=False, optimizer_kwds=None):
        self.BaseModel = BaseModel
        PeriodicModelerMultiband.__init__(self, optimizer,
                                          fit_period=fit_period,
                                          optimizer_kwds=optimizer_kwds)

    def _fit(self, t, y, dy, filts):
        masks = [(filts == f) for f in self.unique_filts_]
        self.models_ = [self.BaseModel().fit(t[m], y[m], dy[m]) for m in masks]

    def _score(self, periods):
        # Total score is the sum of powers weighted by chi2-normalization
        powers = np.array([model.score(periods) for model in self.models_])
        baselines = np.array([model.baseline_err for model in self.models_])
        return np.dot(baselines / baselines.sum(), powers)

    def _predict(self, t, filts, period):
        t, filts = np.broadcast_arrays(t, filts)

        result = np.zeros(t.shape, dtype=float)
        masks = ((filts == f) for f in self.unique_filts_)
        for model, mask in zip(self.models_, masks):
            result[mask] = model.predict(t[mask], period=period)
        return result
