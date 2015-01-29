from numpy.testing import assert_equal
from .. import RRLyraeGenerated, fetch_rrlyrae


def test_rrlyrae_generated():
    rrlyrae = fetch_rrlyrae()
    lcid = rrlyrae.ids[100]

    gen = RRLyraeGenerated(lcid)
    assert_equal(gen.period, rrlyrae.get_metadata(lcid)['P'])

    # smoke test
    for corrected in [True, False]:
        t, y, dy = gen.observed('g', corrected=corrected)
        y = gen.generated('g', t, dy, corrected=corrected)
