from .. import xr
from .. import np
from .. import pd


def gettest():
    import tempfile
    from netCDF4 import Dataset
    with tempfile.TemporaryDirectory() as tmpdirname:
        f = Dataset(f'{tmpdirname}/test.nc', 'w')
        f.createDimension('TSTEP', None)
        f.createDimension('DATE-TIME', 2)
        f.createDimension('LAY', 1)
        f.createDimension('VAR', 1)
        f.createDimension('ROW', 3)
        f.createDimension('COL', 8)
        TFLAG = f.createVariable('TFLAG', 'i', ('TSTEP', 'VAR', 'DATE-TIME'))
        TFLAG.units = "<YYYYDDD,HHMMSS>"
        TFLAG.long_name = "TFLAG           "
        TFLAG.var_desc = (
            "Timestep-valid flags:  (1) YYYYDDD or (2) HHMMSS".ljust(80)
        )
        TA = f.createVariable('TA', 'f', ('TSTEP', 'LAY', 'ROW', 'COL'))
        TA.long_name = "TA              "
        TA.units = "K               "
        TA.var_desc = (
            "Variable TA output from M3 prototype LCM EM".ljust(80)
        )

        f.IOAPI_VERSION = "1.0 1997349 (Dec. 15, 1997)"
        f.EXEC_ID = "????????????????".ljust(80)
        f.FTYPE = 1
        f.CDATE = 2008072
        f.CTIME = 224149
        f.WDATE = 2008072
        f.WTIME = 224149
        f.SDATE = 1983250
        f.STIME = 0
        f.TSTEP = 10000
        f.NTHIK = 1
        f.NCOLS = 8
        f.NROWS = 3
        f.NLAYS = 1
        f.NVARS = 1
        f.GDTYP = 2
        f.P_ALP = 30.
        f.P_BET = 60.
        f.P_GAM = -90.
        f.XCENT = -90.
        f.YCENT = 40.
        f.XORIG = -2831931.
        f.YORIG = -730076.
        f.XCELL = 600000.
        f.YCELL = 600000.
        f.VGTYP = 1
        f.VGTOP = np.float32(10000.)
        f.VGLVLS = np.array([1., 0.98], dtype='f')
        f.GDNAM = "ALPHA_CROSS     "
        f.UPNAM = "                "
        f.setncattr('VAR-LIST', "TA              ")
        f.FILEDESC = ""
        f.HISTORY = ""
        TFLAG[0, 0] = [1983250, 0]

        TA[0] = [
            303.7344, 303.2994, 303.1, 303.2, 303.3, 303.4, 303.5, 303.6,
            303.1111, 303.2222, 303.3, 303.4, 303.5, 303.6, 303.6, 303.7,
            303.7557, 303.4526, 303.6, 303.7, 303.8, 303.9, 303.1, 303.2
        ]
        f.close()
        ds = xr.open_dataset(f'{tmpdirname}/test.nc')
    return ds


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
