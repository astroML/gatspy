import numpy as np
from numpy.testing import assert_allclose, assert_, assert_raises
from nose import SkipTest

from .. import LombScargle, TrendedLombScargle


def _generate_data(N=100, period=1, theta=[10, 2, 3], dy=1, rseed=0):
    """Generate some data for testing"""
    rng = np.random.RandomState(rseed)
    t = 20 * period * rng.rand(N)
    omega = 2 * np.pi / period
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_lomb_scargle_linear_trend(N=100, period=1, slope=5):
    """Test whether the generalized lomb-scargle properly fits data with a
    linear trend component
    """
    t, y, dy = _generate_data(N, period)
    model = LombScargle().fit(t, y, dy)
    model.optimizer.period_range = (period - 0.5, period + 0.5)
    y_hat = model.predict(t)

    y_trend = y + slope * t
    model = TrendedLombScargle().fit(t, y, dy)
    model.optimizer.period_range = (period - 0.5, period + 0.5)
    y_hat_trend = model.fit(t, y_trend, dy).predict(t)
    assert_allclose(y_hat, y_hat_trend - slope * t, rtol=5E-2)
