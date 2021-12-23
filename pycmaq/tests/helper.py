import xarray as xr
import importlib
import numpy as np

try:
    importlib.import_module('shapely')
    has_shapely = True
except Exception:
    has_shapely = False


def gettest(coords=False):
    from .. import open_dataset
    import tempfile
    from netCDF4 import Dataset
    with tempfile.TemporaryDirectory() as tmpdirname:
        # File contents copied from 8x3.ncf made by Todd Plessel
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

        if coords:
            ds = open_dataset(f'{tmpdirname}/test.nc')
        else:
            ds = xr.open_dataset(f'{tmpdirname}/test.nc')
    return ds


def getutc():
    ds = gettest(True)
    ds['HOUR'] = ds['TA'] * 0
    dss = []
    for i in range(24):
        tmpds = ds.copy(deep=True)
        tmpds.coords['TSTEP'] = tmpds.coords['TSTEP'] + np.timedelta64(i * 1)
        tmpds['HOUR'][:] = i
        dss.append(tmpds)

    outds = xr.concat(dss, dim='TSTEP')
    return outds


def getstdatm():
    """
    PV variable is made up for testing.
    """

    ds2d = gettest(True)
    zh = np.array([
        2000, 5000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000,
        16500
    ])
    lvls = np.array([
        1.000, 0.621, 0.358, 0.211, 0.168, 0.131, 0.099, 0.070, 0.046, 0.024,
        0.010, 0.000
    ], dtype='f')
    ds2d.attrs['VGLVLS'] = lvls
    mlvls = (lvls[:-1] + lvls[1:]) / 2
    tfrac = np.array([
        0.953, 0.898, 0.845, 0.835, 0.826, 0.818, 0.811, 0.804, 0.798, 0.792,
        0.790
    ])
    dss = []
    for i, lvl in enumerate(mlvls):
        tmpds = ds2d.copy(deep=True)
        tmpds.coords['LAY'] = np.array([lvl], dtype='f')
        tmpds['TA'] = tmpds['TA'] * tfrac[i]
        tmpds['PRES'] = tmpds['TA'] * 0 + lvl * (101325. - 10000) + 10000
        tmpds['ZH'] = tmpds['TA'] * 0 + zh[i]
        tmpds['PV'] = tmpds['TA'] * 0 + 1 + (
            int(zh[i] > 12000)
            + int(zh[i] > 13000)
            + int(zh[i] > 14000)
            + int(zh[i] > 15000)
            + int(zh[i] > 16000)
        )
        dss.append(tmpds)

    ds = xr.concat(dss, dim='LAY')

    return ds
