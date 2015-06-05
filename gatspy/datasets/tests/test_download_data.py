from .. import (fetch_rrlyrae, fetch_rrlyrae_fitdata,
                fetch_rrlyrae_templates, fetch_rrlyrae_lc_params)
from nose import SkipTest
from numpy.testing import assert_equal

try:
    # Python 3
    from urllib.error import URLError
    ConnectionError = ConnectionResetError
except ImportError:
    # Python 2
    from urllib2 import URLError
    from socket import error as ConnectionError


def test_downloads():
    for downloader in (fetch_rrlyrae, fetch_rrlyrae_fitdata,
                       fetch_rrlyrae_templates, fetch_rrlyrae_lc_params):
        data = downloader()
        assert data is not None


def test_forced_download():
    """Test downloading the smallest of the files: table3.dat.gz (22K)"""
    try:
        data = fetch_rrlyrae_fitdata(force_download=True)
    except (URLError, ConnectionError):
        raise SkipTest("No internet connection: data download test skipped")
    assert_equal(data.shape, (483,))
    assert_equal(data.dtype.names, ('id', 'RA', 'DEC', 'rExt', 'd', 'RGC',
                                    'u', 'g', 'r', 'i', 'z', 'V',
                                    'ugmin', 'ugmin_err', 'grmin', 'grmin_err'))
