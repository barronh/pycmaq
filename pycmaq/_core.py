__all__ = ['xr']
import xarray as xr
import pandas as pd
import numpy as np


@xr.register_dataset_accessor("cmaq")
class CmaqAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._proj4 = None
        self._pyproj = None
        self._pycno = None

    def get_time_fromtflag(self, mid=False):
        from datetime import datetime
        obj = self._obj
        jdates = obj.TFLAG.isel(**{'VAR': 0, 'DATE-TIME': 0}).values
        jdates = np.maximum(jdates, 1001)
        hhmmss = obj.TFLAG.isel(**{'VAR': 0, 'DATE-TIME': 1}).values
        times = np.array([
            datetime.strptime(f'{j:07d} {t:06d}', '%Y%j %H%M%S')
            for j, t in zip(jdates, hhmmss)
        ])
        if mid:
            dt = self.get_timedelta()
            times = times + dt / 2
        times = xr.DataArray(times, dims=('TSTEP'), name='TSTEP')
        return times

    def get_timedelta(self):
        obj = self._obj
        TSTEP = obj.attrs['TSTEP']
        hours = TSTEP // 10000
        minutes = TSTEP % 10000 // 100
        seconds = TSTEP % 100
        dt = pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return dt

    def get_time_frommeta(self, mid=False):
        obj = self._obj
        reftime = pd.to_datetime(f'{obj.SDATE:07d}', format='%Y%j')
        nt = obj.dims['TSTEP']
        dt = self.get_timedelta()
        times = np.array([
            reftime + dt * ti + 0.5 * dt
            for ti in np.arange(nt)
        ])
        times = xr.DataArray(times, dims=('TSTEP'), name='TSTEP')
        return times

    def get_time(self, mid=False):
        obj = self._obj
        if 'TFLAG' in obj.variables:
            return self.get_time_fromtflag(mid=mid)
        else:
            return self.get_time_frommeta(mid=mid)

    def set_time_coord(self, mid=False):
        obj = self._obj
        obj['TSTEP'] = self.get_time(mid=mid)

    @property
    def proj4(self):
        """Return the projection of this dataset."""
        if self._proj4 is None:
            from PseudoNetCDF.coordutil import getproj4
            self._proj4 = getproj4(self._obj, withgrid=True)
        return self._proj4

    @property
    def pyproj(self):
        if self._pyproj is None:
            import pyproj
            self._pyproj = pyproj.Proj(self.proj4, preserve_units=True)
        return self._pyproj

    @property
    def pycno(self):
        if self._pycno is None:
            import pycno
            self._pycno = pycno.cno(proj=self.pyproj)
        return self._pycno

    def drawcno(self, cnopath=None, ax=None, **kwds):
        return self.pycno.draw(cnopath, ax=ax, **kwds)

    def drawcountries(self, ax=None, **kwds):
        return self.drawcno('MWDB_Coasts_Countries_3.cnob', ax=ax, **kwds)

    def drawstates(self, ax=None, **kwds):
        return self.drawcno('MWDB_Coasts_USA_3.cnob', ax=ax, **kwds)

# Consider adding more formal support on the backend
# https://xarray.pydata.org/en/stable/internals/how-to-add-new-backend.html
