from .. import xr
from .. import np
from ..utils import met
from .helper import getstdatm


def test_dz():
    ds = xr.DataArray(
        np.linspace(1, 10, 10)[None, :],
        dims=('TSTEP', 'LAY'),
        name='ZF'
    ).to_dataset()
    DZ = met.dz(ds)
    assert((DZ[:] == 1).all())


def test_pvtroposphere():
    inputf = xr.DataArray(
        np.linspace(0, 10, 11)[None, :],
        dims=('TSTEP', 'LAY'),
        name='PV'
    ).to_dataset()

    refvals = np.zeros(11, dtype='bool')
    # Review >= note in pvtroposphere
    refvals[:2] = True
    chkvals = met.pvtroposphere(inputf, threshold=3).values.ravel()
    assert((refvals == chkvals).all())


def test_wmotroposphere():
    stdf = getstdatm()
    istrop = met.wmotroposphere(stdf)
    refvals = istrop.max('TSTEP').sum('LAY').values
    chkvals = np.ones((3, 8), dtype='i') * 7
    assert((refvals == chkvals).all())


def test_pressure_interp():
    ds = getstdatm()
    lvls = ds.attrs['VGLVLS'][1:3]
    OUTPRES = (
        ds['TA'].isel(LAY=slice(0, 2))
        * 0 + xr.DataArray(
            lvls,
            dims=('LAY',)
        ) * (101325. - 10000) + 10000
    )
    outta = met.pressure_interp(OUTPRES, ds['PRES'], ds['TA'])
    chkvals = np.array([[
        [[279.59699403, 279.19654964, 279.01300238, 279.10506593,
          279.19709896, 279.28914449, 279.38120804, 279.47327158],
         [279.02323166, 279.12549397, 279.19709896, 279.28914449,
          279.38120804, 279.47327158, 279.47327158, 279.56533513],
         [279.61658560, 279.33757337, 279.47327158, 279.56533513,
          279.65735015, 279.74941370, 279.01300238, 279.10506593]],

        [[262.42727848, 262.05142235, 261.87914898, 261.96555600,
          262.05195209, 262.13834817, 262.22475519, 262.31116222],
         [261.88875628, 261.98474010, 262.05195209, 262.13834817,
          262.22475519, 262.31116222, 262.31116222, 262.39756924],
         [262.44567775, 262.18378840, 262.31116222, 262.39756924,
          262.48393481, 262.57034184, 261.87914898, 261.96555600]]
    ]])
    assert(np.allclose(outta.values, chkvals))
