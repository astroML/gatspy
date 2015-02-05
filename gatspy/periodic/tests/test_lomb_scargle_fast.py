from __future__ import division

import numpy as np
from numpy.testing import assert_allclose

from ..lomb_scargle_fast import extirpolate


def test_extirpolate():
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(50)
    y = np.sin(x)
    f = lambda x: np.sin(x / 10)

    x_hat = np.arange(100)
    y_hat = extirpolate(x, y, 100, M=5)

    assert_allclose(np.dot(f(x), y), np.dot(f(x_hat), y_hat))
