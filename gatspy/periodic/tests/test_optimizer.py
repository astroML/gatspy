"""Tests of the various optimizer classes"""
import numpy as np
from numpy.testing import assert_allclose

from ..optimizer import LinearScanOptimizer
from .. import LombScargle


def test_linear_scan():
    optimizer = LinearScanOptimizer(period_range=(0.8, 1.2))

    t = np.linspace(0, 10, 1000)
    y = np.sin(2 * np.pi * t)
    dy = 1

    model = LombScargle(optimizer=optimizer).fit(t, y, dy)

    # test finding best period
    best_period = optimizer.best_period(model)
    assert_allclose(best_period, 1, atol=1E-4)

    # test finding N best periods
    best_periods, best_scores = optimizer.find_best_periods(model, 5,
                                                            return_scores=True)
    assert_allclose(model.score(best_periods), best_scores)
    assert_allclose(best_periods[0], best_period)
