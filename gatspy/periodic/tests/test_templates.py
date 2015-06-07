import numpy as np
from numpy.testing import assert_allclose, assert_raises

from .. import RRLyraeTemplateModeler, RRLyraeTemplateModelerMultiband
from ...datasets import fetch_rrlyrae_templates, fetch_rrlyrae
from scipy.interpolate import UnivariateSpline


def test_basic_template_model():
    template_id = 25

    templates = fetch_rrlyrae_templates()
    phase, y = templates.get_template(templates.ids[template_id])
    model = UnivariateSpline(phase, y, s=0, k=5)

    theta = [17, 0.5, 0.3]
    period = 0.63
    rng = np.random.RandomState(0)
    t = rng.rand(20)
    mag = theta[0] + theta[1] * model((t / period - theta[2]) % 1)

    model = RRLyraeTemplateModeler('ugriz')
    model.fit(t, mag, 1)

    # check that the model matches what we expect
    assert_allclose(model._model(t, theta, period, template_id), mag)

    # check that the optimized model matches the input
    for use_gradient in [True, False]:
        theta_fit = model._optimize(period, template_id, use_gradient)
        assert_allclose(theta, theta_fit, rtol=1E-4)

    # check that the chi2 is near zero
    assert_allclose(model._chi2(theta_fit, period, template_id), 0,
                    atol=1E-8)


def test_multiband_fit():
    # TODO: this is a long test.
    # We could artificially limit the number of templates to make it faster
    rrlyrae = fetch_rrlyrae()
    t, y, dy, filts = rrlyrae.get_lightcurve(rrlyrae.ids[0])
    t = t[::10]
    y = y[::10]
    dy = dy[::10]
    filts = filts[::10]
    period = rrlyrae.get_metadata(rrlyrae.ids[0])['P']

    tfit = np.linspace(0, 5 * period, 99)
    filts_fit = np.array(list('ugriz'))[:, None]

    model = RRLyraeTemplateModelerMultiband().fit(t, y, dy, filts)
    yfit_all = model.predict(tfit, filts_fit, period)

    yfit_band = []
    for filt in 'ugriz':
        mask = (filts == filt)
        model = RRLyraeTemplateModeler(filt)
        model.fit(t[mask], y[mask], dy[mask])
        yfit_band.append(model.predict(tfit, period))

    assert_allclose(yfit_all, yfit_band)


def test_bad_args():
    assert_raises(ValueError, RRLyraeTemplateModeler, filts='abc')
