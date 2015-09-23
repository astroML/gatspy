from __future__ import division
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal

from .. import LombScargle

def test_lombscargle_fap(){
    t = np.linspace(0, 100, 1000)
    np.random.seed(4)
    y = np.random.normal(0, 1.0, t.size)

    fmax = 500/t.max()

    ls = LombScargle().fit(t, y, 1.0)
    ls.optimizer.quiet = True
    ls.optimizer.period_range = (1/fmax, t.max())

    assert_almost_equal(ls.false_alarm_max(), 0.17453, decimal=3)
}
