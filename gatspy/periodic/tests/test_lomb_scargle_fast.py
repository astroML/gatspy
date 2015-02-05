from __future__ import division

import numpy as np
from numpy.testing import assert_allclose

from ..lomb_scargle_fast import extirpolate


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
