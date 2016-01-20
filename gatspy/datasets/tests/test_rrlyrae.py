import pickle
import numpy as np
from numpy.testing import assert_equal, assert_raises
from .. import fetch_rrlyrae, fetch_rrlyrae_templates
from nose import SkipTest

try:
    # Python 3
    from urllib.error import URLError
    ConnectionError = ConnectionResetError
except ImportError:
    # Python 2
    from urllib2 import URLError
    from socket import error as ConnectionError

def test_rrlyrae_lightcurves():
    for partial in [True, False]:
        try:
            rrlyrae = fetch_rrlyrae(partial=partial)
        except(URLError, ConnectionError):
            raise SkipTest("No internet connection: "
                           "data download test skipped")
        lcid = rrlyrae.ids[0]

        if not partial:
            # test the 2D light curve
            t, y, dy = rrlyrae.get_lightcurve(lcid, return_1d=False)
            assert(t.ndim == 2)
            assert(t.shape == y.shape)
            assert(t.shape == dy.shape)

        # test the 1D light curve
        t, y, dy, filts = rrlyrae.get_lightcurve(lcid)
        assert(t.ndim == 1)
        assert(t.shape == y.shape)
        assert(t.shape == dy.shape)
        assert(t.shape == filts.shape)
        assert(not np.any(np.isnan(t + y + dy)))

        # test getting fit metadata
        meta = rrlyrae.get_metadata(lcid)
        assert(meta['id'] == int(lcid))

        # test getting observational metadata
        meta = rrlyrae.get_obsmeta(lcid)
        assert(meta['id'] == int(lcid))


def test_bad_lcid():
    try:
        rrlyrae = fetch_rrlyrae()
    except(URLError, ConnectionError):
        raise SkipTest("No internet connection: "
                       "data download test skipped")
    lcid = 'BAD_ID'

    assert_raises(ValueError, rrlyrae.get_lightcurve, lcid)
    assert_raises(ValueError, rrlyrae.get_metadata, lcid)
    assert_raises(ValueError, rrlyrae.get_obsmeta, lcid)

    try:
        rrlyrae = fetch_rrlyrae(partial=True)
    except(URLError, ConnectionError):
        raise SkipTest("No internet connection: "
                       "data download test skipped")
    assert_raises(ValueError, rrlyrae.get_lightcurve, rrlyrae.ids[0],
                  return_1d=False)


def test_rrlyrae_pickle():
    for partial in [True, False]:
        try:
            rrlyrae = fetch_rrlyrae(partial=partial)
        except(URLError, ConnectionError):
            raise SkipTest("No internet connection: "
                           "data download test skipped")        
        s = pickle.dumps(rrlyrae)
        rrlyrae2 = pickle.loads(s)

        lcid = rrlyrae2.ids[0]
        assert_equal(rrlyrae.get_lightcurve(lcid),
                     rrlyrae2.get_lightcurve(lcid))


def test_rrlyrae_templates():
    try:
        templates = fetch_rrlyrae_templates()
    except(URLError, ConnectionError):
        raise SkipTest("No internet connection: "
                       "data download test skipped")

    filename = templates.filenames[0]
    tid = templates.ids[0]
    t = templates.get_template(tid)
    assert_raises(ValueError, templates.get_template, 'BAD_ID')
