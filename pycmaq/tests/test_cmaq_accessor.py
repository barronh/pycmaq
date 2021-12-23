from .. import xr
from .. import np
from .. import pd
import pytest
from .helper import has_shapely, gettest, getstdatm, getutc


def test_cmaq_accessor():
    ds = gettest()
    assert(hasattr(ds, 'cmaq'))


def test_cmaq_pyproj():
    import pyproj
    ds = gettest()
    inlon = np.arange(-110, -50, 10)
    inlat = np.arange(20, 80, 10)

    refval = pyproj.Proj(
        '+proj=lcc +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-90.0'
        + ' +y_0=730076.0 +x_0=2831931.0 +a=6370000 +b=6370000'
        + ' +to_meter=600000.0 +no_defs', preserve_units=True
    )(inlon, inlat)
    chkval = ds.cmaq.pyproj(inlon, inlat)

    assert(np.allclose(chkval, refval))


def test_cmaq_set_time_coord(ds=None):
    if ds is None:
        ds = gettest()
        ds.cmaq.set_time_coord()
    refval = pd.to_datetime('1983-09-07T00:00:00.000000000')
    assert(
        (ds['TSTEP'] == refval).all()
    )


def test_cmaq_get_time():
    ds = gettest()
    refval = np.array('1983-09-07T00:00:00.0', dtype='datetime64[ns]')
    chkval = ds.cmaq.get_time(tflag=True)
    assert((chkval == refval).all())
    chkval = ds.cmaq.get_time(tflag=False)
    assert((chkval == refval).all())
    refval = np.array('1983-09-07T00:30:00.0', dtype='datetime64[ns]')
    chkval = ds.cmaq.get_time(tflag=True, mid=True)
    assert((chkval == refval).all())
    chkval = ds.cmaq.get_time(tflag=False, mid=True)
    assert((chkval == refval).all())


def test_cmaq_set_vglvls_coord(ds=None):
    if ds is None:
        ds = gettest()
        ds.cmaq.set_vglvls_coord()
    refval = (ds.VGLVLS[1:] + ds.VGLVLS[:-1]) / 2
    assert(
        (ds['LAY'] == refval).all()
    )


def test_cmaq_set_row_coord(ds=None):
    if ds is None:
        ds = gettest()
        ds.cmaq.set_row_coord()
    refval = np.arange(ds.NROWS) + 0.5
    assert(
        (ds['ROW'] == refval).all()
    )


def test_cmaq_set_col_coord(ds=None):
    if ds is None:
        ds = gettest()
        ds.cmaq.set_col_coord()
    refval = np.arange(ds.NCOLS) + 0.5
    assert(
        (ds['COL'] == refval).all()
    )


def test_cmaq_set_coords():
    ds = gettest()
    ds.cmaq.set_coords()
    test_cmaq_set_time_coord(ds)
    test_cmaq_set_vglvls_coord(ds)
    test_cmaq_set_row_coord(ds)
    test_cmaq_set_col_coord(ds)


def test_cmaq_to_ioapi():
    import tempfile
    ds = gettest()
    ds.cmaq.set_coords()
    with tempfile.TemporaryDirectory() as tmpdirname:
        outpath = f'{tmpdirname}/writeout.nc'
        print(0)
        chkds = ds.cmaq.to_ioapi(outpath, close=False)
        print(1)
        for key, chkv in chkds.variables.items():
            print(2)
            if key != 'TFLAG':
                refv = ds[key]
                assert(np.allclose(refv, chkv))


def test_cmaq_to_dataframe():
    ds = gettest()
    ds.cmaq.set_coords()
    df = ds.cmaq.to_dataframe()
    assert(
        np.allclose(ds.variables['TA'][:].values.ravel(), df['TA'].values)
    )


def test_cmaq_from_dataframe():
    ds = gettest()
    ds.cmaq.set_coords()
    df = ds.cmaq.to_dataframe()
    rds = xr.Dataset.cmaq.from_dataframe(df)
    assert(
        np.allclose(ds['TA'][:], rds['TA'][:])
    )


def from_dataframe_incomplete():
    pytest.skip("Needs to be implemented; volunteers?")


def test_cmaq_gridfraction():
    if not has_shapely:
        pytest.skip("requires shapely")
    ds = gettest(True)
    from shapely.geometry import Polygon
    s = Polygon([[0.5, 0.5], [7.5, 0.5], [7.5, 2.5], [0.5, 2.5], [0.5, 0.5]])
    fa = ds.cmaq.gridfraction([s], srcproj=None, propname='area')
    chkvals = np.zeros((ds.NROWS, ds.NCOLS), dtype='f')
    chkvals[:] = 0.5
    chkvals[1:-1, 1:-1] = 1
    chkvals[0, 0] = 0.25
    chkvals[0, -1] = 0.25
    chkvals[-1, -1] = 0.25
    chkvals[-1, 0] = 0.25
    assert(np.allclose(fa, chkvals))


def test_cmaq_to_lst_dataarray():
    utcf = getutc()
    tzvals = np.array([-10, -9, -8, -7, -6, -5, -4, -3])[None, :].repeat(3, 0)
    tz = xr.DataArray(
        tzvals,
        dims=('ROW', 'COL')
    )
    lsthour = utcf.cmaq.to_lst(tz, utcf.HOUR)
    assert((lsthour[0, 0].values == -tz.values).all())


def test_cmaq_to_lst_dataset():
    utcf = getutc()
    tzvals = np.array([-10, -9, -8, -7, -6, -5, -4, -3])[None, :].repeat(3, 0)
    tz = xr.DataArray(
        tzvals,
        dims=('ROW', 'COL')
    )
    lstf = utcf.cmaq.to_lst(tz)
    HOUR = lstf.HOUR
    assert((HOUR[0, 0] == -tz.values).all())
    HOUR[tzvals[0], 0, 0, np.arange(8)]
    assert(np.isnan(HOUR[:, 0, 0].values[tzvals[0], np.arange(8)]).all())
    assert((HOUR[:, 0, 0].values[tzvals[0] - 1, np.arange(8)] == 23).all())


def test_cmaq_findtrop():
    stdf = getstdatm()
    istrop_wmo = stdf.cmaq.findtrop(method='wmo', pvthreshold=2, wmomin=5000)
    assert((istrop_wmo.sum('LAY').values == 7).all())
    istrop_pv2 = stdf.cmaq.findtrop(method='pv', pvthreshold=2)
    assert((istrop_pv2.sum('LAY').values == 5).all())
    istrop_pv3 = stdf.cmaq.findtrop(method='pv', pvthreshold=3)
    assert((istrop_pv3.sum('LAY').values == 6).all())
    istrop_hyand = stdf.cmaq.findtrop(
        method='hybrid', pvthreshold=2, hybrid='and'
    )
    assert((istrop_hyand.sum('LAY').values == 5).all())
    istrop_hyor = stdf.cmaq.findtrop(
        method='hybrid', pvthreshold=2, hybrid='or'
    )
    assert((istrop_hyor.sum('LAY').values == 7).all())


def test_cmaq_pressure_interp():
    ds = getstdatm()
    lvls = ds.attrs['VGLVLS'][1:3]
    OUTPRES = (
        ds['TA'].isel(LAY=slice(0, 2))
        * 0 + xr.DataArray(
            lvls,
            dims=('LAY',)
        ) * (101325. - 10000) + 10000
    )
    outds = ds.cmaq.pressure_interp(OUTPRES, ds['PRES'])
    assert(np.allclose(outds['PRES'].values, OUTPRES.values))
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
    assert(np.allclose(outds['TA'].values, chkvals))
