"""
Implementation of a template modeler for RR Lyrae Stars.

This is based on the algorithm described in Sesar 2010; the same paper is the
source of the templates used here for the fit.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

from ..datasets import fetch_rrlyrae_templates
from .modeler import PeriodicModeler, PeriodicModelerMultiband


__all__ = ['RRLyraeTemplateModeler', 'RRLyraeTemplateModelerMultiband']


class BaseTemplateModeler(PeriodicModeler):
    """
    Base class for single-band template models

    To extend this, overload the ``_template_ids`` and ``_get_template_by_id``
    methods.
    """

    def __init__(self, optimizer=None, fit_period=False, optimizer_kwds=None):
        self.templates = self._build_interpolated_templates()
        if len(self.templates) == 0:
            raise ValueError('No templates available!')
        PeriodicModeler.__init__(self, optimizer=optimizer,
                                 fit_period=fit_period,
                                 optimizer_kwds=optimizer_kwds)

    def _build_interpolated_templates(self):
        self.templates = [self._interpolated_template(tid)
                          for tid in self._template_ids()]
        return self.templates

    def _interpolated_template(self, templateid):
        """Return an interpolator for the given template"""
        phase, y = self._get_template_by_id(templateid)

        # double-check that phase ranges from 0 to 1
        assert phase.min() >= 0
        assert phase.max() <= 1

        # at the start and end points, we need to add ~5 points to make sure
        # the spline & derivatives wrap appropriately
        phase = np.concatenate([phase[-5:] - 1, phase, phase[:5] + 1])
        y = np.concatenate([y[-5:], y, y[:5]])

        # Univariate spline allows for derivatives; use this!
        return UnivariateSpline(phase, y, s=0, k=5)

    def _fit(self, t, y, dy):
        if dy.size == 1:
            ymean = np.mean(y)
        else:
            w = 1 / dy ** 2
            ymean = np.dot(y, w) / w.sum()
        self.chi2_0_ = np.sum((y - ymean) ** 2 / self.dy ** 2)

    def _score(self, periods):
        scores = np.zeros(periods.shape)

        for i, period in enumerate(periods.flat):
            theta_best, chi2 = self._eval_templates(period)
            scores.flat[i] = 1 - min(chi2) / self.chi2_0_

        return scores

    def _predict(self, t, period):
        theta_best, chi2 = self._eval_templates(period)
        i_best = np.argmin(chi2)
        return self._model(t, theta_best[i_best], period, i_best)

    def _eval_templates(self, period):
        """Evaluate the best template for the given period"""
        theta_best = [self._optimize(period, tmpid)
                      for tmpid, _ in enumerate(self.templates)]
        chi2 = [self._chi2(theta, period, tmpid)
                for tmpid, theta in enumerate(theta_best)]

        return theta_best, chi2

    def _model(self, t, theta, period, tmpid):
        """Compute model at t for the given parameters, period, & template"""
        template = self.templates[tmpid]
        phase = (t / period - theta[2]) % 1
        return theta[0] + theta[1] * template(phase)

    def _chi2(self, theta, period, tmpid, return_gradient=False):
        """
        Compute the chi2 for the given parameters, period, & template

        Optionally return the gradient for faster optimization
        """
        template = self.templates[tmpid]
        phase = (self.t / period - theta[2]) % 1
        model = theta[0] + theta[1] * template(phase)
        chi2 = (((model - self.y) / self.dy) ** 2).sum()

        if return_gradient:
            grad = 2 * (model - self.y) / self.dy ** 2
            gradient = np.array([np.sum(grad),
                                 np.sum(grad * template(phase)),
                                 -np.sum(grad * theta[1]
                                         * template.derivative(1)(phase))])
            return chi2, gradient
        else:
            return chi2

    def _optimize(self, period, tmpid, use_gradient=True):
        """Optimize the model for the given period & template"""
        theta_0 = [self.y.min(), self.y.max() - self.y.min(), 0]
        result = minimize(self._chi2, theta_0, jac=bool(use_gradient),
                          bounds=[(None, None), (0, None), (None, None)],
                          args=(period, tmpid, use_gradient))
        return result.x

    #------------------------------------------------------------
    # Overload the following two functions in base classes

    def _template_ids(self):
        """Return the list of template ids"""
        raise NotImplementedError()

    def _get_template_by_id(self, template_id):
        """Get a particular template

        Parameters
        ----------
        template_id : simple type
            Template ID used by base class to define templates

        Returns
        -------
        phase, y : ndarrays
            arrays containing the sorted phase and associated y-values.
        """
        raise NotImplementedError()


class RRLyraeTemplateModeler(BaseTemplateModeler):
    """Template-fitting periods for single-band RR Lyrae

    This class contains functionality to evaluate the fit of the Sesar 2010
    RR Lyrae templates to single-band data.

    Parameters
    ----------
    filts : list or iterable of characters (optional)
        The filters of the templates to be used. Items should be among 'ugriz'.
        Default is 'ugriz'; i.e. all available templates.
    optimizer : PeriodicOptimizer instance (optional)
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
    fit_period : bool (optional)
        If True, then fit for the best period when fit() method is called.
    optimizer_kwds : dict (optional
        Dictionary of keyword arguments for constructing the optimizer

    See Also
    --------
    RRLyraeTemplateModelerMultiband : multiband version of template model
    """
    _raw_templates = None

    def __init__(self, filts='ugriz', optimizer=None,
                 fit_period=False, optimizer_kwds=None):
        self.filts = list(filts)
        self._load_templates()
        BaseTemplateModeler.__init__(self, optimizer=optimizer,
                                     fit_period=fit_period,
                                     optimizer_kwds=optimizer_kwds)

    @classmethod
    def _load_templates(cls):
        """Load lightcurve templates & save as class variable"""
        if cls._raw_templates is None:
            cls._raw_templates = fetch_rrlyrae_templates()

    def _template_ids(self):
        return (tid for tid in self._raw_templates.ids
                if tid[-1] in self.filts)

    def _get_template_by_id(self, tid):
        return self._raw_templates.get_template(tid)


class RRLyraeTemplateModelerMultiband(PeriodicModelerMultiband):
    """Multiband version of RR Lyrae template-fitting modeler

    This class contains functionality to evaluate the fit of the Sesar 2010
    RR Lyrae templates to multiband data.

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance (optional)
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.

    See Also
    --------
    RRLyraeTemplateModeler : single band version of template model
    """

    def _fit(self, t, y, dy, filts):
        self.models_ = []
        for filt in self.unique_filts_:
            mask = (filts == filt)
            model = RRLyraeTemplateModeler(filts=filt)
            model.fit(t[mask], y[mask], dy[mask])
            self.models_.append(model)
        self.modeldict_ = dict(zip(self.unique_filts_, self.models_))

    def _score(self, periods):
        weights = [model.chi2_0_ for model in self.models_]
        scores = [model.score(periods) for model in self.models_]
        return np.dot(weights, scores) / np.sum(weights)

    def _predict(self, t, filts, period):
        result = np.zeros(t.shape)
        for filt in np.unique(filts):
            mask = (filts == filt)
            result[mask] = self.modeldict_[filt].predict(t[mask], period)
        return result
