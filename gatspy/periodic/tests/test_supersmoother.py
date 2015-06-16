from __future__ import division

import numpy as np
from numpy.testing import assert_allclose, assert_, assert_equal, assert_raises
from nose import SkipTest

from .. import SuperSmoother, SuperSmootherMultiband


def _generate_data(N=100, period=1, theta=[10, 2, 3], dy=1, rseed=0):
    """Generate some data for testing"""
    rng = np.random.RandomState(rseed)
    t = 10 * period * rng.rand(N)
    omega = 2 * np.pi / period
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_supersmoother(N=100, period=1):
    t, y, dy = _generate_data(N, period)

    model = SuperSmoother().fit(t, y, dy)
    model.optimizer.period_range = (period / 1.1, period * 1.1)
    model.optimizer.final_pass_coverage = 0
    assert_allclose(period, model.best_period, rtol=0.02)


def test_supersmoother_dy_scalar(N=100, period=1):
    t, y, dy = _generate_data(N, period)

    # Make dy array all the same
    dy[:] = dy.mean()
    periods = np.linspace(period / 2, period * 2, 100)

    assert_equal(SuperSmoother().fit(t, y, dy).score(periods),
                 SuperSmoother().fit(t, y, dy[0]).score(periods))


def test_supersmoother_dy_None(N=100, period=1):
    t, y, dy = _generate_data(N, period)
    periods = np.linspace(period / 2, period * 2, 100)

    assert_equal(SuperSmoother().fit(t, y, 1).score(periods),
                 SuperSmoother().fit(t, y).score(periods))


def test_supersmoother_multiband(N=100, period=1):
    """Test that results are the same with/without filter labels"""
    t, y, dy = _generate_data(N, period)
    periods = np.linspace(period / 2, period * 2, 100)

    model = SuperSmoother()
    P_singleband = model.fit(t, y, dy).score(periods)

    filts = np.ones(N)
    model_mb = SuperSmootherMultiband()
    P_multiband = model_mb.fit(t, y, dy, filts).score(periods)
    assert_allclose(P_multiband, P_singleband)

    tfit = [1.5, 2, 2.5]
    assert_allclose(model.predict(tfit, period=period),
                    model_mb.predict(tfit, 1, period=period))

    assert_raises(ValueError, model_mb.predict, tfit, 2)
