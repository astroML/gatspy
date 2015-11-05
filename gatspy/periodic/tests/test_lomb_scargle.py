import numpy as np
from numpy.testing import assert_allclose, assert_, assert_raises
from nose import SkipTest

from .. import LombScargle, LombScargleAstroML, LombScargleFast


def _generate_data(N=100, period=1, theta=[10, 2, 3], dy=1, rseed=0):
    """Generate some data for testing"""
    rng = np.random.RandomState(rseed)
    t = 20 * period * rng.rand(N)
    omega = 2 * np.pi / period
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_periodogram_auto(N=100, period=1):
    t, y, dy = _generate_data(N, period)
    period, score = LombScargle().fit(t, y, dy).periodogram_auto()

    def check_model(Model):
        p, s = Model().fit(t, y, dy).periodogram_auto()
        assert_allclose(p, period)
        assert_allclose(s, score, atol=1E-2)

    for Model in [LombScargle, LombScargleAstroML, LombScargleFast]:
        yield check_model, Model


def test_lomb_scargle_std_vs_centered(N=100, period=1):
    """Test whether the standard and generalized lomb-scargle
    give close to the same results for non-centered data"""
    t, y, dy = _generate_data(N, period)
    periods = np.linspace(period - 0.5, period + 0.5, 100)

    def check_model(Model):
        P1 = Model(fit_offset=True).fit(t, y, dy).score(periods)
        P2 = Model(fit_offset=False).fit(t, y, dy).score(periods)

        rms = np.sqrt(np.mean((P1 - P2) ** 2))
        assert_(rms < 0.005)

    for Model in [LombScargle, LombScargleAstroML]:
        yield check_model, Model


def test_dy_scalar(N=100, period=1):
    t, y, dy = _generate_data(N, period)

    # Make dy array all the same
    dy[:] = dy.mean()

    def check_model(Model):
        assert_allclose(Model().fit(t, y, dy).periodogram_auto(),
                        Model().fit(t, y, dy[0]).periodogram_auto())

    for Model in [LombScargle, LombScargleAstroML, LombScargleFast]:
        yield check_model, Model


def test_vs_astroML(N=100, period=1):
    t, y, dy = _generate_data(N, period)
    periods = np.linspace(period - 0.5, period + 0.5, 100)

    def compare_models(model1, model2):
        P = [model.fit(t, y, dy).score(periods)
             for model in (model1, model2)]
        assert_allclose(P[0], P[1])

    # standard lomb-scargle
    for fit_offset in [True, False]:
        yield (compare_models,
               LombScargle(fit_offset=fit_offset),
               LombScargleAstroML(fit_offset=fit_offset))
        yield (compare_models,
               LombScargleAstroML(fit_offset=fit_offset),
               LombScargleAstroML(fit_offset=fit_offset, slow_version=True))

    # Sanity check: make sure they work without centering data
    yield (compare_models,
           LombScargleAstroML(center_data=False),
           LombScargle(center_data=False))


def test_construct_X(N=100, period=1):
    """
    Check whether the X array is constructed correctly
    """
    t, y, dy = _generate_data(N, period)

    X = [LombScargle(Nterms=N, fit_offset=False).fit(t, y, dy)
         ._construct_X(period) for N in [1, 2, 3]]
    Y = [LombScargle(Nterms=N, fit_offset=True).fit(t, y, dy)
         ._construct_X(period) for N in [0, 1, 2, 3]]

    for i in range(3):
        assert_allclose(X[i], Y[i + 1][:, 1:])

    for i in range(4):
        assert_allclose(Y[i][:, 0], 1 / dy)

    for i in range(2):
        assert_allclose(X[i], X[i + 1][:, :2 * (i + 1)])


def test_best_params(N=100, period=1):
    """Quick test for whether best params are computed without failure"""
    theta_true = [10, 2, 3]
    dy = 1.0

    t, y, dy = _generate_data(N, period, theta_true, dy)

    for Nterms in [1, 2, 3]:
        for Model in [LombScargle, LombScargleAstroML, LombScargleFast]:
            if Model is not LombScargle:
                model = Model(center_data=False)
            else:
                model = Model(Nterms=Nterms, center_data=False)
            model.fit(t, y, dy)
            theta_best = model._best_params(2 * np.pi / period)
            assert_allclose(theta_true, theta_best[:3], atol=0.2)


def test_regularized(N=100, period=1):
    theta_true = [10, 2, 3]
    dy = 1.0
    t, y, dy = _generate_data(N, period, theta_true, dy)

    for regularize_by_trace in [True, False]:
        model = LombScargle(Nterms=1, regularization=0.1,
                            regularize_by_trace=regularize_by_trace)
        model.fit(t, y, dy)
        model.optimizer.period_range = (0.2, 1.0)
        pred = model.predict(period)


def test_predict_center_data(N=100, period=1):
    """Test that results are the same for centered and non-centered data"""
    theta_true = [10, 2, 3]
    dy = 1.0
    t, y, dy = _generate_data(N, period, theta_true, dy)
    y += 5

    m1 = LombScargle(center_data=True).fit(t, y, dy).predict(t, period=period)
    m2 = LombScargle(center_data=False).fit(t, y, dy).predict(t, period=period)

    assert_allclose(m1, m2)


def test_bad_args():
    assert_raises(ValueError, LombScargle, Nterms=-2)
    assert_raises(ValueError, LombScargle, Nterms=0, fit_offset=False)
    assert_raises(ValueError, LombScargleAstroML, Nterms=2)
    assert_raises(ValueError, LombScargleFast, Nterms=2)


def test_dy_None(N=100, period=1):
    t, y, dy = _generate_data(N, period)
    period, score = LombScargle().fit(t, y, 1).periodogram_auto()

    def check_model(Model):
        p, s = Model().fit(t, y).periodogram_auto()
        assert_allclose(p, period)
        assert_allclose(s, score, atol=1E-2)

    for Model in [LombScargle, LombScargleAstroML, LombScargleFast]:
        yield check_model, Model


def test_power_normalization(N=100, period=1):
    t, y, dy = _generate_data(N, period)

    def check_model(Model, fit_offset, center_data):
        if (Model is LombScargleAstroML and not (fit_offset or center_data)):
            raise SkipTest("Current AstroML gets this one wrong.")
        model = Model(fit_offset=fit_offset, center_data=center_data)
        model.fit(t, y, dy)
        p, s = model.periodogram_auto()
        assert_(np.all((s >= 0) & (s <= 1)))

    for Model in [LombScargle, LombScargleAstroML, LombScargleFast]:
        for fit_offset in [True, False]:
            for center_data in [True, False]:
                yield check_model, Model, fit_offset, center_data
