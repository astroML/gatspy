import numpy as np
from numpy.testing import assert_allclose, assert_, assert_equal
from nose import SkipTest

from .. import SuperSmoother


def _generate_data(N=100, omega=10, theta=[10, 2, 3], dy=1, rseed=0):
    """Generate some data for testing"""
    rng = np.random.RandomState(rseed)
    t = 10 * (2 * np.pi / omega) * rng.rand(N)
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_supersmoother(N=100, omega=10):
    """Test whether the standard and generalized lomb-scargle
    give close to the same results for non-centered data"""
    t, y, dy = _generate_data(N, omega)
    omegas = np.linspace(1, omega + 1, 100)
    periods = 2 * np.pi / omegas

    model = SuperSmoother().fit(t, y, dy)
    model.optimizer.period_range = (2 * np.pi / (omega + 1),
                                    2 * np.pi / (omega - 1))
    model.optimizer.final_pass_coverage = 0
    assert_allclose(omega, 2 * np.pi / model.best_period, rtol=0.02)


def test_dy_scalar(N=100, omega=10):
    """Test whether the standard and generalized lomb-scargle
    give close to the same results for non-centered data"""
    t, y, dy = _generate_data(N, omega)

    # Make dy array all the same
    dy[:] = dy.mean()
    omegas = np.linspace(1, omega + 1, 10)
    periods = 2 * np.pi / omegas

    assert_equal(SuperSmoother().fit(t, y, dy).score(periods),
                 SuperSmoother().fit(t, y, dy[0]).score(periods))
