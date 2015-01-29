from __future__ import division, print_function

import numpy as np

from .modeler import PeriodicModelerMultiband
from .lomb_scargle import LombScargle, LeastSquaresMixin


class LombScargleMultiband(LeastSquaresMixin, PeriodicModelerMultiband):
    """Multiband Periodogram Implementation

    This implements the generalized multi-band periodogram described in
    VanderPlas & Ivezic 2015.

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

    See Also
    --------
    LombScargle
    LombScargleAstroML
    LombScargleMultibandFast
    """
    def __init__(self, optimizer=None, Nterms_base=1, Nterms_band=1,
                 reg_base=None, reg_band=1E-6, regularize_by_trace=True,
                 center_data=True):
        self.Nterms_base = Nterms_base
        self.Nterms_band = Nterms_band
        self.reg_base = reg_base
        self.reg_band = reg_band
        self.regularize_by_trace = regularize_by_trace
        self.center_data = center_data
        PeriodicModelerMultiband.__init__(self, optimizer)

    def _fit(self, t, y, dy, filts):
        self.unique_filts_ = np.unique(filts)
        self.ymean_ = self._compute_ymean()

        masks = [(filts == filt) for filt in self.unique_filts_]
        self.ymean_by_filt_ = [LeastSquaresMixin._compute_ymean(self,
                                                                y=y[mask],
                                                                dy=dy[mask])
                               for mask in masks]

        self.yw_ = self._construct_y(weighted=True)
        self.regularization = self._construct_regularization()
        return self

    def _construct_regularization(self):
        if self.reg_base is None and self.reg_band is None:
            reg = 0
        else:
            Nbase = 1 + 2 * self.Nterms_base
            Nband = 1 + 2 * self.Nterms_band
            reg = np.zeros(Nbase + len(self.unique_filts_) * Nband)
            if self.reg_base is not None:
                reg[:Nbase] = self.reg_base
            if self.reg_band is not None:
                reg[Nbase:] = self.reg_band
        return reg

    def _compute_ymean(self, **kwargs):
        y = kwargs.get('y', self.y)
        dy = kwargs.get('dy', self.dy)
        filts = kwargs.get('filts', self.filts)

        ymean = np.zeros(y.shape)
        for filt in np.unique(filts):
            mask = (filts == filt)
            ymean[mask] = LeastSquaresMixin._compute_ymean(self, y=y[mask],
                                                           dy=dy[mask])
        return ymean

    def _construct_y(self, weighted=True, **kwargs):
        y = kwargs.get('y', self.y)
        dy = kwargs.get('dy', self.dy)
        filts = kwargs.get('filts', self.filts)
        center_data = kwargs.get('center_data', self.center_data)

        ymean = self._compute_ymean(**kwargs)

        if center_data:
            y = y - ymean

        if weighted:
            return y / dy
        else:
            return y

    def _construct_X(self, omega, weighted=True, **kwargs):
        t = kwargs.get('t', self.t)
        dy = kwargs.get('dy', self.dy)
        filts = kwargs.get('filts', self.filts)

        # X is a huge-ass matrix that has lots of zeros depending on
        # which filters are present...
        #
        # huge-ass, quantitatively speaking, is of shape
        #  [len(t), (1 + 2 * Nterms_base + nfilts * (1 + 2 * Nterms_band))]

        # TODO: the building of the matrix could be more efficient
        cols = [np.ones(len(t))]
        cols = sum(([np.sin((i + 1) * omega * t),
                     np.cos((i + 1) * omega * t)]
                    for i in range(self.Nterms_base)), cols)

        for filt in self.unique_filts_:
            cols.append(np.ones(len(t)))
            cols = sum(([np.sin((i + 1) * omega * t),
                         np.cos((i + 1) * omega * t)]
                        for i in range(self.Nterms_band)), cols)
            mask = (filts == filt)
            for i in range(-1 - 2 * self.Nterms_band, 0):
                cols[i][~mask] = 0

        if weighted:
            return np.transpose(np.vstack(cols) / dy)
        else:
            return np.transpose(np.vstack(cols))

    def _predict(self, t, filts, period):
        vals = set(np.unique(filts))
        if not vals.issubset(self.unique_filts_):
            raise ValueError("filts does not match training data: "
                             "input: {0} output: {1}"
                             "".format(set(self.unique_filts_), set(vals)))

        t, filts = np.broadcast_arrays(t, filts)
        output_shape = t.shape

        t = t.ravel()
        filts = filts.ravel()
        omega = 2 * np.pi / period

        # TODO: broadcast this
        ymeans = np.zeros(len(filts))
        for i, filt in enumerate(filts):
            j = np.where(self.unique_filts_ == filt)[0][0]
            ymeans[i] = self.ymean_by_filt_[j]

        theta = self._best_params(omega)
        X = self._construct_X(omega, weighted=False, t=t, filts=filts)
        return (ymeans + np.dot(X, theta)).reshape(output_shape)


class LombScargleMultibandFast(PeriodicModelerMultiband):
    """Fast Multiband Periodogram Implementation

    This implements the specialized multi-band periodogram described in
    VanderPlas & Ivezic 2015 (with Nbase=0 and Nband=1) which is essentially
    a weighted sum of the standard periodograms for each band.

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
    Nterms : integer (default = 1)
        Number of fourier terms to use in the model
    BaseModel : class type (default = LombScargle)
        The base model to use for each individual band.

    See Also
    --------
    LombScargle
    LombScargleAstroML
    LombScargleMultiband
    """
    def __init__(self, optimizer=None, Nterms=1, BaseModel=LombScargle):
        # Note: center_data must be True, or else the chi^2 weighting will fail
        self.Nterms = Nterms
        self.BaseModel = BaseModel
        PeriodicModelerMultiband.__init__(self, optimizer)

    def _fit(self, t, y, dy, filts):
        self.unique_filts_ = np.unique(filts)
        masks = [(filts == f) for f in self.unique_filts_]
        self.models_ = [self.BaseModel(Nterms=self.Nterms, center_data=True,
                                       fit_offset=True).fit(t[m], y[m], dy[m])
                        for m in masks]

    def _score(self, periods):
        # Total score is the sum of powers weighted by chi2-normalization
        powers = np.array([model.score(periods) for model in self.models_])
        chi2_0 = np.array([np.sum(model.yw_ ** 2) for model in self.models_])
        return np.dot(chi2_0 / chi2_0.sum(), powers)

    def _best_params(self, omega):
        return np.asarray([model._best_params(omega)
                           for model in self.models_])

    def _predict(self, t, filts, period):
        vals = set(np.unique(filts))
        if not vals.issubset(self.unique_filts_):
            raise ValueError("filts does not match training data: "
                             "input: {0} output: {1}"
                             "".format(set(self.unique_filts_), set(vals)))

        t, filts = np.broadcast_arrays(t, filts)

        result = np.zeros(t.shape, dtype=float)
        masks = ((filts == f) for f in self.unique_filts_)
        for model, mask in zip(self.models_, masks):
            result[mask] = model.predict(t[mask], period=period)
        return result
