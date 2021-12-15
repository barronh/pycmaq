from .. import xr


def test_cmaq_accessor():
    ds = xr.open_dataset('../../TestData/MCIP/METCRO2D_20110101')
    assert(hasattr(ds, 'cmaq'))
