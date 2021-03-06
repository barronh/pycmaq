__all__ = ['CmaqAccessor', 'xr', 'pd', 'np', 'plt']

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import utils
import warnings


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
        offset : time delta
            Add offset to times before returning

        Returns
        -------
        time: xarray.DataArray
            times with dimension TSTEP
        """
        obj = self._obj
        jdates = obj.TFLAG.isel(**{'VAR': 0, 'DATE-TIME': 0}).values
        jdates = np.maximum(jdates, 1970001)
        hhmmss = obj.TFLAG.isel(**{'VAR': 0, 'DATE-TIME': 1}).values
        times = np.array([
            pd.to_datetime(f'{j:07d} {t:06d}', format='%Y%j %H%M%S')
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
        offset : time delta
            Add offset to times before returning

        Returns
        -------
        time: xarray.DataArray
            times with dimension TSTEP
        """
        obj = self._obj
        SDATE = obj.SDATE
        if SDATE == -635:
            warnings.warn(
                f'{SDATE} cannot be used to return a time; using 1970001'
            )
            SDATE = 1970001
        reftime = pd.to_datetime(
            f'{SDATE:07d} {obj.STIME:06d}', format='%Y%j %H%M%S'
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
        if obj.attrs.get('GDTYP', 2) == 1:
            obj['COL'] = xr.DataArray(
                (
                    np.arange(obj.NCOLS) * obj.attrs['XCELL']
                    + obj.attrs['XCELL'] / 2 + obj.attrs['XORIG']
                ),
                dims=('COL',), name='COL'
            )
        else:
            obj['COL'] = xr.DataArray(
                np.arange(obj.NCOLS) + 0.5,
                dims=('COL',), name='COL'
            )

    def set_row_coord(self):
        """
        Set ROW coord to mid points 0.5 ... NROWS - 0.5
        """
        obj = self._obj
        if obj.attrs.get('GDTYP', 2) == 1:
            obj['ROW'] = xr.DataArray(
                (
                    np.arange(obj.NROWS) * obj.attrs['YCELL']
                    + obj.attrs['YCELL'] / 2 + obj.attrs['YORIG']
                ),
                dims=('ROW',), name='ROW'
            )
        else:
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
        """
        Returns a list of variables with IOAPI dimensions. TSTEP, LAY
        and either PERIM or ROW and COL.
        """
        from collections import OrderedDict
        obj = self._obj
        OUTVARS = OrderedDict()
        for k, v in obj.variables.items():
            if (
                v.dims == ('TSTEP', 'LAY', 'ROW', 'COL')
                or v.dims == ('TSTEP', 'LAY', 'PERIM')
            ):
                OUTVARS[k] = obj[k]
        return OUTVARS

    def updated_attrs_from_coords(self):
        """
        Returns updated attrs dictionary by adjusting projection/time
        information
        """
        obj = self._obj
        attrs = obj.attrs.copy()
        attrs['NLAYS'] = np.int32(obj.dims['LAY'])
        if 'PERIM' in obj.dims:
            attrs['NCOLS'] = np.int32(obj.attrs['NCOLS'])
            attrs['NROWS'] = np.int32(obj.attrs['NROWS'])
            attrs['XORIG'] = obj.attrs['XORIG']
            attrs['YORIG'] = obj.attrs['YORIG']
        else:
            attrs['NCOLS'] = np.int32(obj.dims['COL'])
            attrs['NROWS'] = np.int32(obj.dims['ROW'])
            attrs['XORIG'] = (
                obj.XORIG + (obj['COL'].values[0] - 0.5) * obj.XCELL
            )
            attrs['YORIG'] = (
                obj.YORIG + (obj['ROW'].values[0] - 0.5) * obj.YCELL
            )
        OUTVARS = self.get_ioapi_variables()
        attrs['NVARS'] = np.int32(len(OUTVARS))
        attrs['VAR-LIST'] = ''.join([k.ljust(16) for k in OUTVARS])
        start_date = obj['TSTEP'].isel(TSTEP=0)
        getpart = start_date.dt.strftime
        attrs['SDATE'] = getpart('%Y%j').values[...].astype('i4')
        attrs['STIME'] = getpart('%H%M%S').values[...].astype('i4')
        return attrs

    def update_from_coords(self):
        """
        Updates attrs dictionary by adjusting projection/time
        information
        """
        attrs = self.updated_attrs_from_coords()
        self._obj.attrs = attrs

    def to_ioapi(
        self, path, mode='w', overwrite=False, close=True, var_kw=None,
        format='NETCDF4_CLASSIC', **kwds
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

        attrs = self.updated_attrs_from_coords()
        times = self.get_time_frommeta(mid=False, offset=None)
        outf = netCDF4.Dataset(path, mode=mode, format=format, **kwds)
        fileattrs = {pk: pv for pk, pv in attrs.items()}
        for pk, pv in fileattrs.items():
            if pk == 'VGTOP':
                pv = np.float32(pv)
            elif pk == 'VGLVLS':
                pv = np.array(pv, dtype='f')
            elif isinstance(pv, (int, np.int64, np.int32)):
                pv = np.int32(pv)

            setattr(outf, pk, pv)

        outf.createDimension('TSTEP', None)
        outf.createDimension('DATE-TIME', 2)
        outf.createDimension('LAY', outf.NLAYS)
        outf.createDimension('VAR', outf.NVARS)

        OUTVARS = self.get_ioapi_variables()
        if 'PERIM' in self._obj.dims:
            outf.createDimension('PERIM', self._obj.dims['PERIM'])
        else:
            outf.createDimension('ROW', outf.NROWS)
            outf.createDimension('COL', outf.NCOLS)
        tflag = outf.createVariable(
            'TFLAG', 'i', ('TSTEP', 'VAR', 'DATE-TIME')
        )
        tflag.long_name = 'TFLAG'.ljust(16)
        tflag.var_desc = 'TFLAG'.ljust(80)
        tflag.units = '<YYYYJJJ,HHMMSS>'.ljust(16)

        for key, var in OUTVARS.items():
            ovar = outf.createVariable(
                key, var.dtype.char, tuple(var.dims), **var_kw
            )
            outattrs = {pk: pv for pk, pv in var.attrs.items()}
            outattrs.setdefault('long_name', key)
            outattrs['long_name'] = outattrs['long_name'].ljust(16)[:16]
            outattrs.setdefault('var_desc', key)
            outattrs['var_desc'] = outattrs['var_desc'].ljust(80)[:80]
            outattrs.setdefault('units', 'unknown')
            outattrs['units'] = outattrs['units'].ljust(16)[:16]

            for pk, pv in outattrs.items():
                ovar.setncattr(pk, pv)
            ovar[:] = var[:]

        if fileattrs['TSTEP'] == 0:
            tflag[:, :, 0] = 0
            tflag[:, :, 1] = 0
        else:
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
        """Return a pyproj.Proj object based on the proj4 attribute."""
        if self._pyproj is None:
            import pyproj
            self._pyproj = pyproj.Proj(self.proj4, preserve_units=True)
        return self._pyproj

    def xy2ll(self, x=None, y=None):
        """
        Convert x/y projected coordinates to longitude and latitude in decimal
        degrees. Simple wrapper around pyproj(x, y, inverse=True)
        """
        if x is None or y is None:
            if x is None and y is None:
                y, x = xr.broadcast(self._obj.ROW, self._obj.COL)
            else:
                raise ValueError('x and y must be none if either is none.')
        lon = x.copy()
        lat = y.copy()
        lon[:], lat[:] = self.pyproj(x.values, y.values, inverse=True)
        return lon, lat

    def ll2xy(self, lon, lat):
        """
        Convert longitude and latitude in decimal degrees to the x/y
        coordinates. Simple wrapper around pyproj(lon, lat, inverse=False).
        """
        x = lon.copy()
        y = lat.copy()
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        x[:], y[:] = self.pyproj(lon, lat, inverse=False)
        return x, y

    def ll2ij(self, lon, lat):
        """
        Convert longitude and latitude in decimal degrees to the row/col
        indices.
        """
        x, y = self.ll2xy(lon, lat)
        col = np.floor(x) + 0.5
        row = np.floor(y) + 0.5
        return col, row

    def perimxy(self):
        """
        Perimeter is defined as the set of cells that goes around the domain.

        It starts at the cell south of the southwest corner (-0.5, 0.5) and
        moves southeast of the southeast corner (-0.5, NCOLS+0.5). From there,
        it goes to the northeast of the northeast corner (NROWS+0.5,
        NCOLS+0.5). From there, it skips to the cell northwest of the northwest
        corner (NROWS+0.5, -0.5) and proceeds to the cell north of the
        northeast corner (NROWS+0.5, NCOLS-0.5). From there, it skips to the
        southwest of the southwest corner (-0.5, -0.5) and goes to the west of
        the northwest corner (NROWS-0.5, -0.5).

        https://www.cmascenter.org/ioapi/documentation/all_versions/html/BDY.jpg
        """
        obj = self._obj
        bx = np.arange(obj.attrs['NCOLS'] + 1) + 0.5
        by = bx * 0 - 0.5
        ry = np.arange(obj.attrs['NROWS'] + 1) + 0.5
        rx = ry * 0 + obj.attrs['NCOLS'] + 0.5
        tx = np.arange(-1, obj.attrs['NCOLS']) + 0.5
        ty = tx * 0 + obj.attrs['NROWS'] + 0.5
        ly = np.arange(-1, obj.attrs['NROWS']) + 0.5
        lx = ly * 0 - 0.5
        px = xr.DataArray(
            np.concatenate([bx, rx, tx, lx]),
            dims=('PERIM',)
        )
        py = xr.DataArray(
            np.concatenate([by, ry, ty, ly]),
            dims=('PERIM',)
        )
        return px, py

    def perimlonlat(self):
        """
        Converts perimxy results to longitude/latitude in decimal degrees.
        """
        px, py = self.perimxy()
        return self.xy2ll(px, py)

    @property
    def pycno(self):
        """Return a pycno.cno object based on the pyproj attribute."""
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

    def to_lst_dataarray(
        self, timezone, var, out=None, verbose=0
    ):
        """
        Arguments
        ---------
        timezone : xr.DataArray
            timezone is the UTC offset in hours (e.g., EST=-5) and should
            have dims (ROW, and COL). If dims LAY and TSTEP are present,
            the first will be used for each.
        var : xr.Variable
            variable to shift to LST
        out : xr.DataArray
            variable to house the output. If None, then out = var * 0

        Returns
        -------
        out : xr.DataArray
            variable with data in LST. Note that all dates are tz-naive
        """
        if out is None:
            out = var.copy()

        if 'TSTEP' in timezone.dims:
            nt = dict(zip(timezone.dims, timezone.shape))['TSTEP']
            assert(nt == 1)
            tzvar = timezone.isel(TSTEP=0).round(0).astype('i')
        else:
            tzvar = timezone.round(0).astype('i')

        if 'LAY' in tzvar.dims:
            nk = dict(zip(tzvar.dims, tzvar.shape))['LAY']
            assert(nk == 1)
            tzlay = tzvar.isel(LAY=0)
        else:
            tzlay = tzvar

        utz = np.unique(tzlay.data)
        nshifted = 0
        ntoshift = np.prod(out.shape[-2:])

        # Using temp numpy array because all other methods of indexing
        # do not correctly align
        temp = np.ma.masked_all(out.shape)

        for tz in utz:
            ridx, cidx = np.where(tzlay.data == tz)
            if verbose > 1:
                print(
                    f'{tz}, {ridx.size}, {ridx.size / ntoshift:6.1%}'
                    + f', {nshifted / ntoshift:6.1%}'
                )

            nshifted += ridx.size
            temp[:, :, ridx, cidx] = var.isel(
                ROW=xr.DataArray(ridx, dims=('out',)),
                COL=xr.DataArray(cidx, dims=('out',))
            ).shift(TSTEP=tz)

        out[:] = temp[:]
        if verbose > 0:
            print('shifted', nshifted, 'of', ntoshift)

        # nk = self._obj.dims['LAY']
        # nj = self._obj.dims['ROW']
        # ni = self._obj.dims['COL']
        #
        # can be substantially optimized.
        # For example, all shifts are k-independent. Remove k loop
        # Also, there are only a few time zones, so shifts can be done on
        # groups of cells
        #
        # Currently, this is a very slow process and should be addressed.
        # for ki in range(nk):
        #     varlay = var.isel(LAY=ki)
        #     for ji in range(nj):
        #         varrow = varlay.isel(ROW=ji)
        #         tzrow = tzlay.isel(ROW=ji)
        #         for ii in range(ni):
        #             varcol = varrow.isel(COL=ii)
        #             tzcol = tzrow.isel(COL=ii).values[...]
        #             out[:, ki, ji, ii] = varcol.shift(TSTEP=-tzcol)
        return out

    def to_lst_dataset(self, timezone, ds=None, verbose=0):
        """
        Arguments
        ---------
        timezone : xr.DataArray
            timezone is the UTC offset in hours (e.g., EST=-5) and should
            have dims (ROW, and COL). If dims LAY and TSTEP are present,
            the first will be used for each.
        ds : xr.Dataset
            File with variables to shift to LST. If None, then self is used

        Returns
        -------
        out : xr.DataArray
            variable with data in LST. Note that all dates are tz-naive
        """

        if ds is None:
            ds = self._obj
        outds = ds.copy(deep=True)
        for varkey, var in outds.variables.items():
            if var.dims == ('TSTEP', 'LAY', 'ROW', 'COL'):
                if verbose > 0:
                    print(varkey, flush=True)
                invar = ds[varkey]
                self.to_lst_dataarray(
                    timezone, invar, out=var, verbose=verbose
                )
        return outds

    def to_lst(self, timezone, oth=None, verbose=0):
        """
        Arguments
        ---------
        timezone : xr.DataArray
            timezone is the UTC offset in hours (e.g., EST=-5) and should
            have dims (ROW, and COL). If dims LAY and TSTEP are present,
            the first will be used for each.
        oth : xr.Variable or xr.Dataset
            variable or dataset to shift to LST. If xr.Variable or
            xr.DataArray, then to_lst_datarray is called. If xr.Dataset, then
            to_lst_dataset is called.
        verbose : int
            Level of verbosity. Passed to to_lst_dataset.

        Returns
        -------
        out : xr.DataArray or xr.Dataset
            variable or dataset with data in LST. Note that all dates are
            tz-naive
        """

        if isinstance(oth, (xr.DataArray, xr.Variable)):
            return self.to_lst_dataarray(
                timezone, var=oth, verbose=verbose
            )
        else:
            return self.to_lst_dataset(
                timezone, ds=oth, verbose=verbose
            )

    def findtrop(self, method=None, pvthreshold=2, wmomin=5000, hybrid='and'):
        """
        Thin wrapper around pycmaq.utils.met functions pvtroposphere and
        wmotroposphere. See their documentation for more information.

        If file has PV, TA, and ZH, it will use a hybrid approach (either)
        If file has PV, it will use the pv approach
        If file has TA and ZH, it will use the WMO approach
        """
        obj = self._obj
        if method is None:
            if (
                'PV' in obj.variables
                and 'TA' in obj.variables
                and 'ZH' in obj.variables
            ):
                method = 'hybrid'
            elif 'PV' in obj.variables:
                method = 'pv'
            elif (
                'TA' in obj.variables
                and 'ZH' in obj.variables
            ):
                method = 'wmo'
        method = method.lower()
        if method == 'pv':
            return utils.met.pvtroposphere(obj, threshold=pvthreshold)
        elif method == 'wmo':
            return utils.met.wmotroposphere(obj, minval=wmomin)
        elif method == 'hybrid':
            ispv = utils.met.pvtroposphere(obj, threshold=pvthreshold)
            iswmo = utils.met.wmotroposphere(obj, minval=wmomin)
            if hybrid.lower() == 'or':
                return ispv | iswmo
            elif hybrid.lower() == 'and':
                return ispv & iswmo
            else:
                raise KeyError(f'{hybrid} unknown; try "or" or "and"')
        else:
            raise KeyError(f'{method} unknown; try "pv", "wmo", or "hybrid"')

    def calcdz(self):
        """
        See pycmaq.utils.met.dz
        """
        return utils.met.dz(self._obj)

    def pressure_interp(
        self, PRESOUT, PRESIN=None, VAR=None, verbose=0, **kwds
    ):
        """
        PRESOUT: xarray.DataArray
            Target file pressure coordinates.
        PRESIN : xarray.DataArray or None
            If provided, this is the pressure for for the source dataset.
            If not provided, the dataset must contain PRES (i.e, MCIP pressure)
        VAR : xarray.DataArray or None
            Interpolate a single variable.
        verbose : int
            Level of verbosity
        kwds : mappable
            Passed pycmaq.utils.met.pressure_interp

        See pycmaq.utils.met.perssure_interp for more details
        """
        from collections import OrderedDict
        obj = self._obj
        if PRESIN is None:
            PRESIN = obj.PRES

        if VAR is not None:
            return utils.met.pressure_interp(
                PRESOUT, PRESIN, VAR, verbose=verbose, **kwds
            )
        else:
            invars = self.get_ioapi_variables()
            outvars = OrderedDict()
            for key, var in invars.items():
                if verbose > 0:
                    print(key)
                outvars[key] = utils.met.pressure_interp(
                    PRESOUT, PRESIN, var, verbose=verbose, **kwds
                )
            outds = xr.Dataset(outvars)
            outds.attrs['VGLVLS'] = np.arange(outds.dims['LAY'] + 1)
            outds.attrs['NLAYS'] = outds.dims['LAY']
            outds.attrs['VGTOP'] = -9999
            return outds

    def gridfraction(
        self, shapes, srcproj=None, clip=True, propname='area', verbose=0
    ):
        """
        See utils.shapes.gridfraction_frompoly
        """
        return utils.shapes.gridfraction_frompoly(
            self._obj, shapes, srcproj=srcproj, clip=True, propname=propname,
            verbose=verbose
        )

    def attr_from_shapefile(
        self, shapepath, key, method='laf', srcproj=None, clip=True,
        units='unknown', verbose=0
    ):
        """
        Calculate timezone offset based on largest area overlap of time zone
        polygons from a shapefile. The offset is returned as a variable.

        Arguments
        ---------
        self : xr.Dataset
            Must support the cmaq accessor function gridfraction
        shapepath : str
            Path to shapefile with timezone as a UTC offset
        key : str
            Attribute field name for UTC offset
        method : str
            Method for selecting shape for each grid cell. Currently supports
            laf, areaweigthed, centroid
        srcproj : scalar
            Projection definition in terms that pyproj can interpret
        clip : bool
            Clip source
        units : str
            Unit attribute of output variable
        verbose : int
            verbosity level.

        Returns
        -------
        tzvar : xr.DataArray
            Array (TSTEP, LAY, ROW, COL) with UTC offsets as 32-bit floats
        """
        return utils.shapes.attr_from_shapefile(
            gf=self._obj, shapepath=shapepath, key=key, method=method,
            srcproj=srcproj, clip=clip, units=units, verbose=verbose
        )

    def to_dataframe(self):
        """
        Thin wrapper around Dataset.to_dataframe:
        * ensures dim_order persists (TSTEP, LAY, ROW, COL PERIM)
        * adds all Dataset.attrs to the dataframe.attrs
        """
        if 'PERIM' in self._obj.dims:
            outdims = ('TSTEP', 'LAY', 'PERIM')
        else:
            outdims = ('TSTEP', 'LAY', 'ROW', 'COL')

        # Only output IOAPI compliant variables
        outds = self._obj[sorted(self.get_ioapi_variables())]
        outdf = outds.to_dataframe(dim_order=outdims)
        outdf.attrs.update(self.updated_attrs_from_coords())
        return outdf

    @classmethod
    def from_dataframe(cls, df, fill_value=None, **ioapi_kw):
        """
        Expects df.attrs to contain meaningful ioapi attributes. These
        attributes can be overwritten by ioapi_kw
        """
        if 'PERIM' in df.index.names:
            indims = ('TSTEP', 'LAY', 'PERIM')
        else:
            indims = ('TSTEP', 'LAY', 'ROW', 'COL')
        outds = xr.Dataset.from_dataframe(df.reorder_levels(indims))
        if fill_value is not None:
            outds = outds.fillna(fill_value)
        outds.attrs.update(df.attrs)
        outds.attrs.update(ioapi_kw)
        return outds

    @classmethod
    def from_dataframe_incomplete(cls, df, fill_value=None, **ioapi_kw):
        from copy import deepcopy
        attrs = deepcopy(df.attrs)
        attrs.update(ioapi_kw)
        sdate = pd.to_datetime(
            f'{attrs["SDATE"]:07d} {attrs["STIME"]:06d}', format='%Y%j %H%M%S'
        )
        dt = (
            pd.to_datetime('1900-01-01 010000', format='%Y-%m-%d %H%M%S')
            - pd.to_datetime('1900-01-01')
        )
        times = pd.date_range(
            sdate, df.index.get_level_values('TSTEP').max(), freq=dt
        )
        ROWS = np.arange(attrs['NROWS']) + 0.5
        COLS = np.arange(attrs['NCOLS']) + 0.5
        LAYS = (attrs['VGLVLS'][:-1] + attrs['VGLVLS'][1:]) / 2
        # outidx = pd.MultiIndex.from_product(
        #     [times, LAYS, ROWS, COLS], names=('TSTEP', 'LAY', 'ROW', 'COL')
        # )
        # fulldf = df.reindex(outidx)
        smallds = cls.from_dataframe(df, fill_value=fill_value, **ioapi_kw)
        outds = smallds.reindex(
            TSTEP=times, COL=COLS, ROW=ROWS, LAY=LAYS,
            fill_value=fill_value
        )
        return outds

# Consider adding more formal support on the backend
# https://xarray.pydata.org/en/stable/internals/how-to-add-new-backend.html
