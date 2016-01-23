from __future__ import division
import warnings

import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
                           assert_warns, assert_no_warnings)

from ..lomb_scargle_fast import (extirpolate, bitceil, trig_sum,
                                 lomb_scargle_fast)
from .. import LombScargle, LombScargleFast


def _generate_data(N=100, period=1, theta=[10, 2, 3], dy=1, rseed=0):
    """Generate some data for testing"""
    rng = np.random.RandomState(rseed)
    t = 20 * period * rng.rand(N)
    omega = 2 * np.pi / period
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_extirpolate():
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(50)
    y = np.sin(x)
    f = lambda x: np.sin(x / 10)

    def check_result(N, M=5):
        y_hat = extirpolate(x, y, N, M)
        x_hat = np.arange(len(y_hat))
        assert_allclose(np.dot(f(x), y), np.dot(f(x_hat), y_hat))

    for N in [100, None]:
        yield check_result, N


def test_extirpolate_with_integers():
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(50)
    x[:25] = x[:25].astype(int)
    y = np.sin(x)
    f = lambda x: np.sin(x / 10)

    def check_result(N, M=5):
        y_hat = extirpolate(x, y, N, M)
        x_hat = np.arange(len(y_hat))
        assert_allclose(np.dot(f(x), y), np.dot(f(x_hat), y_hat))

    for N in [100, None]:
        yield check_result, N


def test_bitceil():
    slow_bitceil = lambda N: int(2 ** np.ceil(np.log2(N)))

    for N in (2 ** np.arange(1, 12)):
        for offset in (-1, 0, 1):
            assert_equal(slow_bitceil(N + offset), bitceil(N + offset))


def test_trig_sum():
    rng = np.random.RandomState(0)
    t = 10 * rng.rand(50)
    h = np.sin(t)

    def check_result(f0, adjust_t, freq_factor, df=0.01):
        tfit = t - t.min() if adjust_t else t
        S1, C1 = trig_sum(tfit, h, df, N=1000, use_fft=True,
                          f0=f0, freq_factor=freq_factor, oversampling=10)
        S2, C2 = trig_sum(tfit, h, df, N=1000, use_fft=False,
                          f0=f0, freq_factor=freq_factor, oversampling=10)
        assert_allclose(S1, S2, atol=1E-2)
        assert_allclose(C1, C2, atol=1E-2)

    for f0 in [0, 1]:
        for adjust_t in [True, False]:
            for freq_factor in [1, 2]:
                yield check_result, f0, adjust_t, freq_factor


def test_lomb_scargle_fast():
    rng = np.random.RandomState(0)

    t = 30 * rng.rand(100)
    y = np.sin(t)
    dy = 0.1 + 0.1 * rng.rand(len(t))
    y += dy * rng.randn(len(t))

    def check_results(center_data, fit_offset):
        freq1, P1 = lomb_scargle_fast(t, y, dy,
                                      center_data=center_data,
                                      fit_offset=fit_offset, use_fft=True)
        freq2, P2 = lomb_scargle_fast(t, y, dy,
                                      center_data=center_data,
                                      fit_offset=fit_offset, use_fft=False)
        assert_allclose(freq1, freq2)
        assert_allclose(P1, P2, atol=0.005)

    for center_data in [True, False]:
        for fit_offset in [True, False]:
            yield check_results, center_data, fit_offset


def test_find_best_period():
    t, y, dy = _generate_data()

    def check_result(use_fft, fit_offset, center_data):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = LombScargle(fit_offset=fit_offset,
                                center_data=center_data).fit(t, y, dy)
        fast = LombScargleFast(fit_offset=fit_offset,
                               center_data=center_data,
                               use_fft=use_fft).fit(t, y, dy)
        model.optimizer.period_range = (0.2, 1.0)
        fast.optimizer.period_range = (0.2, 1.0)
        assert_allclose(model.best_period, fast.best_period, atol=5E-4)

    for use_fft in [True, False]:
        for fit_offset in [True, False]:
            for center_data in [True, False]:
                yield check_result, use_fft, fit_offset, center_data


def test_power():
    t, y, dy = _generate_data()
    f0 = 0.8
    df = 0.01
    N = 40

    def check_result(use_fft, fit_offset, center_data):
        model = LombScargle(fit_offset=fit_offset,
                            center_data=center_data).fit(t, y, dy)
        fast = LombScargleFast(fit_offset=fit_offset,
                               center_data=center_data,
                               use_fft=use_fft).fit(t, y, dy)
        assert_allclose(model.score_frequency_grid(f0, df, N),
                        fast.score_frequency_grid(f0, df, N),
                        atol=0.005)

    for use_fft in [True, False]:
        for fit_offset in [True, False]:
            for center_data in [True, False]:
                yield check_result, use_fft, fit_offset, center_data

def check_warn_on_small_data():
    t, y, dy = _generate_data(20)
    model = LombScargleFast()
    assert_warns(UserWarning, model.score_frequency_grid, 0.8, 0.01, 40)
    model = LombScargleFast(silence_warnings=True)
    assert_no_warnings(model.score_frequency_grid, 0.8, 0.01, 40)
