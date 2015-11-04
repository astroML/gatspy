from __future__ import division

import numpy as np
from numpy.testing import assert_allclose, assert_
from nose import SkipTest

from .. import (LombScargle, LombScargleMultiband, LombScargleMultibandFast)


def _generate_data(N=100, period=1, theta=[10, 2, 3], dy=1, rseed=0):
    """Generate some data for testing"""
    rng = np.random.RandomState(rseed)
    t = 20 * period * rng.rand(N)
    omega = 2 * np.pi / period
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_lomb_scargle_multiband(N=100, period=1):
    """Test that results are the same with/without filter labels"""
    t, y, dy = _generate_data(N, period)
    periods = np.linspace(period / 2, period * 2, 100)
    omega = 2 * np.pi / period

    model = LombScargle(center_data=False, fit_offset=True)
    model.fit(t, y, dy)

    model_mb = LombScargleMultiband(center_data=False)
    model_mb.fit(t, y, dy, filts=1)

    assert_allclose(model.score(periods),
                    model_mb.score(periods))

    assert_allclose(model._best_params(omega),
                    model_mb._best_params(omega)[:3])

    assert_allclose(model.predict(t, period=period),
                    model_mb.predict(t, filts=1, period=period))


def test_multiband_predict(N=100, period=1):
    t, y, dy = _generate_data(N, period)

    rng = np.random.RandomState(0)
    filts = rng.randint(0, 3, N)
    masks = [(filts == f) for f in range(3)]

    tfit = np.linspace(5 * period, 15 * period, 30)
    filtsfit = np.arange(3)[:, None]

    def check_models(Nterms):
        models = [LombScargle(Nterms=Nterms).fit(t[mask], y[mask], dy[mask])
                  for mask in masks]
        model_mb = LombScargleMultiband(Nterms_base=0,
                                        Nterms_band=Nterms)
        model_mb.fit(t, y, dy, filts)

        single_results = [model.predict(tfit, period=period)
                          for model in models]
        multi_results = model_mb.predict(tfit, filtsfit, period=period)

        assert_allclose(single_results, multi_results, rtol=1E-4)

    for Nterms in [1, 2, 3]:
        yield check_models, Nterms


def test_multiband_predict_broadcast(N=100, period=1):
    t, y, dy = _generate_data(3 * N, period)

    t = t.reshape(3, N)
    y = y.reshape(3, N)
    dy = dy.reshape(3, N)

    rng = np.random.RandomState(0)
    filts = np.arange(3)[:, None]

    tfit = np.linspace(5 * period, 15 * period, 30)
    filtsfit = np.arange(3)[:, None]

    def check_models(Nterms):
        models = [LombScargle(Nterms=Nterms).fit(t[i], y[i], dy[i])
                  for i in range(3)]
        model_mb = LombScargleMultiband(Nterms_base=0,
                                        Nterms_band=Nterms)
        model_mb.fit(t, y, dy, filts)

        single_results = [model.predict(tfit, period=period)
                          for model in models]
        multi_results = model_mb.predict(tfit, filtsfit, period=period)

        assert_allclose(single_results, multi_results, rtol=1E-4)

    for Nterms in [1, 2, 3]:
        yield check_models, Nterms


def test_multiband_predict_center_data(N=100, period=1):
    """Test that results are the same for centered and non-centered data"""
    t, y, dy = _generate_data(N, period)

    rng = np.random.RandomState(0)
    filts = rng.randint(0, 3, N)
    masks = [(filts == f) for f in range(3)]

    model1 = LombScargleMultiband(center_data=True).fit(t, y, dy, filts)
    model2 = LombScargleMultiband(center_data=False).fit(t, y, dy, filts)

    tfit = np.linspace(5 * period, 15 * period, 30)
    filtsfit = np.arange(3)[:, None]

    assert_allclose(model1.predict(tfit, filtsfit, period=period),
                    model2.predict(tfit, filtsfit, period=period),
                    rtol=1E-6)
