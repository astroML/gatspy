import numpy as np
from numpy.testing import assert_allclose

from ..modeler import PeriodicModeler, PeriodicModelerMultiband
from .. import LombScargle, LombScargleAstroML, SuperSmoother
from .. import LombScargleMultiband, LombScargleMultibandFast, NaiveMultiband


class FakeModeler(PeriodicModeler):
    """Fake periodic modeler for testing PeriodicModeler base"""
    def _fit(self, t, y, dy):
        pass

    def _predict(self, t, period):
        return np.ones(len(t))

    def _score(self, periods):
        return np.exp(-np.abs(periods - np.round(periods)))


class FakeModelerMultiband(PeriodicModelerMultiband):
    """Fake periodic modeler for testing PeriodicModelerMultiband base"""
    def _fit(self, t, y, dy, filts):
        pass

    def _predict(self, t, filts, period):
        return np.ones(len(t))

    def _score(self, periods):
        return np.exp(-np.abs(periods - np.round(periods)))


def test_modeler_base():
    """Smoke-test of PeriodicModeler base class"""
    t = np.linspace(0, 10, 100)
    y = np.random.rand(len(t))
    dy = 0.1

    model = FakeModeler()

    # test setting the period range for the optimizer
    model.optimizer.period_range = (0.8, 1.2)

    # test fitting the model
    model.fit(t, y, dy)

    # test the score function
    assert_allclose(model.score(1.0), 1.0)

    # test the best_period property
    assert_allclose(model.best_period, 1, rtol=1E-4)

    # test the predict function
    assert_allclose(model.predict(t), 1)


def test_modeler_base_multiband():
    """Smoke-test of PeriodicModelerMultiband base class"""
    t = np.linspace(0, 10, 100)
    y = np.random.rand(len(t))
    dy = 0.1
    filts = None

    model = FakeModelerMultiband()

    # test setting the period range for the optimizer
    model.optimizer.period_range = (0.8, 1.2)

    # test fitting the model
    model.fit(t, y, dy, filts)

    # test the score function
    assert_allclose(model.score(1.0), 1.0)

    # test the best_period property
    assert_allclose(model.best_period, 1, rtol=1E-4)

    # test the predict function
    assert_allclose(model.predict(t, filts), 1)


def _generate_data(N=100, omega=10, theta=[10, 2, 3], dy=1, rseed=0):
    """Generate some data for testing"""
    rng = np.random.RandomState(rseed)
    t = 20 * (2 * np.pi / omega) * rng.rand(N)
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_modelers_smoketest():
    t, y, dy = _generate_data()
    periods = np.linspace(0.2, 1.0, 5)

    def check_model(Model):
        model = Model()
        model.fit(t, y, dy)

        # Make optimization fast
        model.optimizer.period_range = (0.5, 0.52)
        model.optimizer.final_pass_coverage = 0
        model.best_period

        model.score(periods)
        model.predict(t)

    for Model in (LombScargle, LombScargleAstroML, SuperSmoother):
        yield check_model, Model


def test_multiband_modelers_smoketest():
    t, y, dy = _generate_data()
    periods = np.linspace(0.2, 1.0, 5)
    filts = np.arange(len(t)) % 3

    def check_model(Model):
        model = Model()
        model.fit(t, y, dy, filts)

        # Make optimization fast
        model.optimizer.period_range = (0.5, 0.52)
        model.optimizer.final_pass_coverage = 0
        period = model.best_period

        model.predict(t, filts)

    for Model in (LombScargleMultiband, NaiveMultiband,
                  LombScargleMultibandFast):
        yield check_model, Model
