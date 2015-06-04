"""
Implementation of a template modeler for RR Lyrae Stars
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy import optimize

from ..datasets import fetch_rrlyrae_templates
from .modeler import PeriodicModeler, PeriodicModelerMultiband


def build_interpolated_templates(filts='ugriz'):
    templates = fetch_rrlyrae_templates()
    ids = templates.ids

    template_dict = dict((band, {}) for band in filts)

    for tmpid in templates.ids:
        band = tmpid[-1]
        if band not in filts:
            continue

        phase, y = templates.get_template(tmpid)

        # explicitly add phase=1 to avoid extrapolation
        phase = np.concatenate([phase, [1]])
        y = np.concatenate([y, y[:1]])
        template_dict[band][tmpid[:-1]] = interp1d(phase, y)

    return template_dict


class RRLyraeTemplateModeler(PeriodicModeler):
    """Template-fitting periods for single-band RR Lyrae"""

    raw_templates = fetch_rrlyrae_templates()

    @classmethod
    def _interpolated_template(cls, template_id, interpolater=interp1d):
        phase, y = cls.raw_templates.get_template(template_id)

        # explicitly add phase=1 to avoid extrapolation
        phase = np.concatenate([phase, [1]])
        y = np.concatenate([y, y[:1]])

        return interpolater(phase, y)

    def __init__(self, filts='ugriz', optimizer=None):
        print(filts)
        self.templates = [self._interpolated_template(tmpid)
                          for tmpid in self.raw_templates.ids
                          if tmpid[-1] in filts]
        assert len(self.templates) > 0
        PeriodicModeler.__init__(self, optimizer)

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
        theta_best = [self._optimize(period, tmpid)
                      for tmpid, _ in enumerate(self.templates)]
        chi2 = [self._chi2(theta, period, tmpid)
                for tmpid, theta in enumerate(theta_best)]

        return theta_best, chi2

    def _model(self, t, theta, period, tmpid):
        template = self.templates[tmpid]
        phase = (t / period - theta[2]) % 1
        return theta[0] + theta[1] * template(phase)
        
    def _chi2(self, theta, period, tmpid):
        return ((self._model(self.t, theta, period, tmpid) - self.y) ** 2
                / self.dy ** 2).sum()
    
    def _optimize(self, period, tmpid):
        theta_0 = [self.y.min(), self.y.max() - self.y.min(), 0]
        return optimize.fmin(self._chi2, theta_0,
                             args=(period, tmpid),
                             disp=False)


class RRLyraeTemplateModelerMultiband(PeriodicModelerMultiband):
    """Multiband version of RR Lyrae template-fitting modeler"""

    def _fit(self, t, y, dy, filts):
        self.models_ = []
        for filt in self.unique_filts_:
            mask = (filts == filt)
            model = RRLyraeTemplateModeler(filts=filt).fit(t[mask],
                                                           y[mask],
                                                           dy[mask])
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

