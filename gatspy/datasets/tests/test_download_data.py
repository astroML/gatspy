from .. import (fetch_rrlyrae, fetch_rrlyrae_fitdata,
                fetch_rrlyrae_templates, fetch_rrlyrae_lc_params)


def test_downloads():
    for downloader in (fetch_rrlyrae, fetch_rrlyrae_fitdata,
                       fetch_rrlyrae_templates, fetch_rrlyrae_lc_params):
        data = downloader()
        assert data is not None
                
