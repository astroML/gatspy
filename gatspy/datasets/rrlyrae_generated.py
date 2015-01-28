"""Tools to generate light curves"""

import numpy as np
from scipy.interpolate import interp1d
from . import fetch_rrlyrae_templates, fetch_rrlyrae


class RRLyraeGenerated(object):
    """Generate RR Lyrae light curves from Sesar 2010 lightcurves

    Parameters
    ----------
    lcid : int
        Valid light curve ID from the Sesar 2010 RR Lyrae dataset
    random_state : int
        Random number generator seed

    Attributes
    ----------
    lcdata : RRLyraeLC object
        Container for the RR Lyrae light curve dataset
    templates : RRLyraeTemplates object
        Container for the RR Lyrae template dataset
    period : float
        Period of the RR Lyrae object
    """
    lcdata = fetch_rrlyrae()
    templates = fetch_rrlyrae_templates()

    # Extinction corrections: Table 1 from Berry et al. (2012, ApJ, 757, 166).
    ext_correction = {'u': 1.810,
                      'g': 1.400,
                      'r': 1.0,
                      'i': 0.759,
                      'z': 0.561 }
    
    @classmethod
    def _template_func(cls, num, band, mu=0, A=1):
        template_id = "{0:.0f}{1}".format(num, band)
    
        phase, amp = cls.templates.get_template(template_id)
        phase = np.concatenate([phase, [1]])
        amp = np.concatenate([amp, amp[-1:]])

        return interp1d(phase, mu + A * amp)
    
    def __init__(self, lcid, random_state=None):
        self.lcid = lcid
        self.meta = self.lcdata.get_metadata(lcid)
        self.obsmeta = self.lcdata.get_obsmeta(lcid)
        self.rng = np.random.RandomState(random_state)
        
    @property
    def period(self):
        return self.meta['P']
    
    def observed(self, band, corrected=True):
        if band not in 'ugriz':
            raise ValueError("band='{0}' not recognized".format(band))
        i = 'ugriz'.find(band)
        t, y, dy = self.lcdata.get_lightcurve(self.lcid, return_1d=False)

        if corrected:
            ext = self.obsmeta['rExt'] * self.ext_correction[band]
        else:
            ext = 0

        return t[:, i], y[:, i] - ext, dy[:, i]

    def generated(self, band, t, err=None, corrected=True):
        t = np.asarray(t)
        num = self.meta[band + 'T']
        mu = self.meta[band + '0']
        amp = self.meta[band + 'A']
        t0 = self.meta[band + 'E']

        if corrected:
            ext = 0
        else:
            ext = self.obsmeta['rExt'] * self.ext_correction[band]
        
        func = self._template_func(num, band, mu + ext, amp)
        mag = func(((t - t0) / self.period) % 1)
        
        if err is not None:
            mag += self.rng.normal(0, err, t.shape)
            
        return mag
