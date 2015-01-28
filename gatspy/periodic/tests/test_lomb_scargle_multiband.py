import numpy as np
from numpy.testing import assert_allclose, assert_
from nose import SkipTest

from .. import (LombScargle, LombScargleMultiband, LombScargleMultibandFast)


def _generate_data(N=100, omega=10, theta=[10, 2, 3], dy=1, rseed=0):
    """Generate some data for testing"""
    rng = np.random.RandomState(rseed)
    t = 20 * (2 * np.pi / omega) * rng.rand(N)
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_lomb_scargle_multiband(N=100, omega=10):
    """Test that results are the same with/without filter labels"""
    t, y, dy = _generate_data(N, omega)
    omegas = np.linspace(1, omega + 1, 100)
    periods = 2 * np.pi / omegas

    model = LombScargle(center_data=False, fit_offset=True)
    P_singleband = model.fit(t, y, dy).score(periods)

    filts = np.ones(N)
    model_mb = LombScargleMultiband(center_data=False)
    P_multiband = model_mb.fit(t, y, dy, filts).score(periods)
    assert_allclose(P_multiband, P_singleband)
    assert_allclose(model._best_params(omega),
                    model_mb._best_params(omega)[:3])
