__all__ = ['xr', 'pd', 'np', 'plt']

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib as plt


@xr.register_dataset_accessor("cmaq")
class CmaqAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._proj4 = None
        self._pyproj = None
        self._pycno = None

    def get_timedelta(self):
        obj = self._obj
        TSTEP = obj.attrs['TSTEP']
        hours = TSTEP // 10000
        minutes = TSTEP % 10000 // 100
        seconds = TSTEP % 100
        dt = pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return dt

    def get_time_fromtflag(self, mid=False, offset=None):
        """
        Derive time from TFLAG. This can take a while if the file is buffered
        and has many times and variables.

        Arguments
        ---------
        mid : bool
            If mid, assume the true time is half a time step ahead of TFLAG.

        Returns
        -------
        time: xarray.DataArray
            times with dimension TSTEP
        """
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
        if offset is not None:
            times = times + offset
        return times

    def get_time_frommeta(self, mid=False, offset=None):
        """
        Derive time from SDATE and STIME. This is faster than from TFLAG, but
        in theory, the TFLAG could differ.

        Arguments
        ---------
        mid : bool
            If mid, assume the true time is half a time step ahead of TFLAG.

        Returns
        -------
        time: xarray.DataArray
            times with dimension TSTEP
        """
        obj = self._obj
        reftime = pd.to_datetime(
            f'{obj.SDATE:07d} {obj.STIME:06d}', format='%Y%j %H%M%S'
        )
        nt = obj.dims['TSTEP']
        dt = self.get_timedelta()
        times = np.array([
            reftime + dt * ti
            for ti in np.arange(nt)
        ])
        if mid:
            times = times + dt / 2

        times = xr.DataArray(times, dims=('TSTEP'), name='TSTEP')
        if offset is not None:
            times = times + offset
        return times

    def get_time(self, mid=False, tflag=None, offset=None):
        """
        Derive time from TFLAG or SDATE/STIME.

        Arguments
        ---------
        mid : bool
            If mid, assume the true time is half a time step ahead of TFLAG.
        tflag : None or bool
            If is None, then choose method based on TFLAG availability.
            If True, use TFLAG
            If False, use SDATE/STIME

        Returns
        -------
        time: xarray.DataArray
            times with dimension TSTEP
        """

        obj = self._obj
        if tflag is None:
            tflag = 'TFLAG' in obj.variables

        if tflag:
            return self.get_time_fromtflag(mid=mid, offset=offset)
        else:
            return self.get_time_frommeta(mid=mid, offset=offset)

    def set_col_coord(self):
        """
        Set COL coord to mid points 0.5 ... NCOLS - 0.5
        """
        obj = self._obj
        obj['COL'] = xr.DataArray(
            np.arange(obj.NCOLS) + 0.5,
            dims=('COL',), name='COL'
        )

    def set_row_coord(self):
        """
        Set ROW coord to mid points 0.5 ... NROWS - 0.5
        """
        obj = self._obj
        obj['ROW'] = xr.DataArray(
            np.arange(obj.NROWS) + 0.5,
            dims=('ROW',), name='ROW'
        )

    def set_vglvls_coord(self):
        """
        Set LAY coord to mid points of VGLVLS
        """
        obj = self._obj
        obj['LAY'] = xr.DataArray(
            (obj.VGLVLS[1:] + obj.VGLVLS[:-1]) / 2,
            dims=('LAY',), name='LAY'
        )

    def set_time_coord(self, mid=False, tflag=None, offset=None):
        """
        Set TSTEP coord based. See get_time_coord for keyword details.
        """
        obj = self._obj
        obj['TSTEP'] = self.get_time(mid=mid, tflag=tflag, offset=offset)

    def set_coords(self, timemid=False, tflag=None, offset=None):
        """
        equivalent to
            set_time_coord(mid=timemid, tflag=tflag)
            set_vglvls_coord()
            set_row_coord()
            set_col_coord()
        """
        self.set_time_coord(mid=timemid, tflag=tflag, offset=offset)
        self.set_vglvls_coord()
        self.set_row_coord()
        self.set_col_coord()
    def get_ioapi_variables(self):
        from collections import OrderedDict
        obj = self._obj
        OUTVARS = OrderedDict()
        for k, v in obj.variables.items():
            if (
                v.dims == ('TSTEP', 'LAY', 'ROW', 'COL')
                or v.dims == ('TSTEP', 'LAY', 'PERIM')
            ):
                OUTVARS[k] = v
        return OUTVARS

    def updated_attrs_from_coords(self):
        obj = self._obj
        attrs = obj.attrs.copy()
        attrs['NCOLS'] = np.int32(obj.dims['COL'])
        attrs['NROWS'] = np.int32(obj.dims['ROW'])
        attrs['NLAYS'] = np.int32(obj.dims['LAY'])
        attrs['XORIG'] = obj.XORIG + (obj['COL'].values[0] - 0.5) * obj.XCELL
        attrs['YORIG'] = obj.YORIG + (obj['ROW'].values[0] - 0.5) * obj.YCELL
        OUTVARS = self.get_ioapi_variables()
        attrs['NVARS'] = np.int32(len(OUTVARS))
        attrs['VAR-LIST'] = ''.join([k.ljust(16) for k in OUTVARS])
        start_date = obj['TSTEP'].isel(TSTEP=0)
        attrs['SDATE'] = start_date.dt.strftime('%Y%j').values[...].astype('i4')
        attrs['STIME'] = start_date.dt.strftime('%H%M%S').values[...].astype('i4')
        return attrs

    def update_from_coords(self):
        attrs = self.updated_attrs_from_coords()
        self._obj.attrs.update(**attrs)

    def to_ioapi(
        self, path, mode='w', overwrite=False, close=True, var_kw=None, **kwds
    ):
        """
        Arguments
        ---------
        path : str
            path for output
        mode : str
            how to open the file ('w', 'ws', 'a+', ...)
        overwrite : bool
            If the file exists, should it try to overwrite?
        close : bool
            After writing, should the file be closed. If True, the file is
            closed and None is returned. If False, the file is returned in
            its open state.
        var_kw : dict
            Passed as key words to output variables. Useful for zlib=True and
            complevel=1-9.
        kwds : dict
            Passed when opening the file

        Returns
        -------
        out : None or file
            See close keyword
        """
        import os
        import netCDF4

        if os.path.exists(path) and not overwrite:
            raise IOError('File exists. Remove or set overwrite=True')

        if var_kw is None:
            var_kw = {}
        obj = self._obj
        attrs = self.updated_attrs_from_coords()
        times = self.get_time_frommeta(mid=False, offset=None)
        outf = netCDF4.Dataset(path, mode=mode, **kwds)
        for pk, pv in attrs.items():
            if pk == 'VGTOP':
                pv = np.float32(pv)
            elif pk == 'VGLVLS':
                pv = np.array(pv, dtype='f')
            
            setattr(outf, pk, pv)

        outf.createDimension('TSTEP', None)
        outf.createDimension('DATE-TIME', 2)
        outf.createDimension('LAY', outf.NLAYS)
        outf.createDimension('VAR', outf.NVARS)
        outf.createDimension('ROW', outf.NROWS)
        outf.createDimension('COL', outf.NCOLS)
        tflag = outf.createVariable('TFLAG', 'i', ('TSTEP', 'VAR', 'DATE-TIME'))
        tflag.long_name = 'TFLAG'.ljust(16)
        tflag.var_desc = 'TFLAG'.ljust(80)
        tflag.units = '<YYYYJJJ,HHMMSS>'.ljust(16)

        OUTVARS = self.get_ioapi_variables()
        for key, var in OUTVARS.items():
            ovar = outf.createVariable(key, var.dtype.char, tuple(var.dims), **var_kw)
            for pk, pv in var.attrs.items():
                ovar.setncattr(pk, pv)
            ovar[:] = var[:]
        
        tflag[:, :, 0] = times.dt.strftime('%Y%j').astype('i')
        tflag[:, :, 1] = times.dt.strftime('%H%M%S').astype('i')
        if close:
            outf.close()
            return None
        else:
            return outf

    @property
    def proj4(self):
        """Return the projection of this dataset."""
        if self._proj4 is None:
            from PseudoNetCDF.coordutil import getproj4
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
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

    def cnodraw(self, cnopath=None, ax=None, **kwds):
        """
        Thin wrapper on pycno for easy access to drawing
        """
        return self.pycno.draw(cnopath, ax=ax, **kwds)

    def cnocountries(self, ax=None, **kwds):
        """
        Thin wrapper on pycno for easy access to drawing
        MWDB_Coasts_Countries_3.cnob
        """
        return self.cnodraw('MWDB_Coasts_Countries_3.cnob', ax=ax, **kwds)

    def cnostates(self, ax=None, **kwds):
        """
        Thin wrapper on pycno for easy access to drawing MWDB_Coasts_USA_3.cnob
        """
        return self.cnodraw('MWDB_Coasts_USA_3.cnob', ax=ax, **kwds)

    def to_lst_dataarray(self, timezone, var, out=None):
        if out is None:
            out = var * 0
        nk = self._obj.dims['LAY']
        nj = self._obj.dims['ROW']
        ni = self._obj.dims['COL']
        if 'TSTEP' in timezone.dims:
            assert(timezone.dims['TSTEP'] == 1)
            tzvar = timezone.isel(TSTEP=0).round(0).astype('i')
        else:
            tzvar = timezone.round(0).astype('i')

        # can be substantially optimized.
        # For example, all shifts are k-independent. Remove k loop
        # Also, there are only a few time zones, so shifts can be done on groups of cells
        for ki in range(nk):
            varlay = var.isel(LAY=ki)
            if 'LAY' in tzvar.dims:
                tzlay = tzvar.isel(LAY=ki)
            else:
                tzlay = tzvar
            lasttz = 0
            for ji in range(nj):
                varrow = varlay.isel(ROW=ji)
                tzrow = tzlay.isel(ROW=ji)
                for ii in range(ni):
                    varcol = varrow.isel(COL=ii)
                    tzcol = tzrow.isel(COL=ii).values[...]                        
                    out[:, ki, ji, ii] = varcol.shift(TSTEP=-tzcol)
        return out

    def to_lst_dataset(self, timezone, ds=None, verbose=0):
        if ds is None:
            ds = self._obj
        outds = ds.copy()
        for varkey, var in outds.variables.items():
            if var.dims == ('TSTEP', 'LAY', 'ROW', 'COL'):
                if verbose > 0:
                    print(varkey, flush=True)
                invar = ds[varkey]
                self.to_lst_dataarray(timezone, invar, out=var)
        return outds

    def to_lst(self, timezone, oth=None, verbose=0):
        if isinstance(oth, (xr.DataArray, xr.Variable)):
            return self.to_lst_dataarray(timezone, var=oth)
        else:
            return self.to_lst_dataset(timezone, ds=oth, verbose=verbose)

# Consider adding more formal support on the backend
# https://xarray.pydata.org/en/stable/internals/how-to-add-new-backend.html
