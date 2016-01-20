from numpy.testing import assert_equal
from .. import RRLyraeGenerated, fetch_rrlyrae

from nose import SkipTest

try:
    # Python 3
    from urllib.error import URLError
    ConnectionError = ConnectionResetError
except ImportError:
    # Python 2
    from urllib2 import URLError
    from socket import error as ConnectionError


def test_rrlyrae_generated():
    try:
        rrlyrae = fetch_rrlyrae()
    except(URLError, ConnectionError):
        raise SkipTest("No internet connection: "
                       "data download test skipped")

    lcid = rrlyrae.ids[100]

    gen = RRLyraeGenerated(lcid)
    assert_equal(gen.period, rrlyrae.get_metadata(lcid)['P'])

    # smoke test
    for corrected in [True, False]:
        t, y, dy = gen.observed('g', corrected=corrected)
        y = gen.generated('g', t, dy, corrected=corrected)
