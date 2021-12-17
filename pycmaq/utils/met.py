__all__ = ['dz', 'pvtroposphere', 'wmotroposphere', 'pressure_interp']

import numpy as np


def dz(inputf):
    ZF = inputf['ZF']
    DZ = ZF.copy()
    DZ[:, 1:] = ZF.diff('LAY')
    DZ.name = 'DZ'
    return DZ


def pvtroposphere(inputf, threshold=2):
    istrop = (
        (
            inputf['PV'] < threshold
        ).isel(LAY=slice(None, None, -1)).cumsum('LAY') > 1
    ).isel(LAY=slice(None, None, -1))
    return istrop


def wmotroposphere(inputf, minval=5000):
    """
    Arguments
    ---------
    ZH : array
        ZH from METCRO3D in meters with the vertical dimension in the second
        position dims(time, lay, row, col)
    TA : array
        TA from METCRO3D in Kelvin with the vertical dimension in the second
        position dims(time, lay, row, col)
    minval : scalar
        Minimum height in meters of the tropopause

    Returns
    -------
    istrop : array
        Booleans that are true if the cell is in the troposphere


    Examples
    --------
    > m3f = Dataset('METCRO3D')
    > TA = m3f.variables['TA']
    > ZH = m3f.variables['ZH']
    > istrop = wmotroposphere(ZH, TA)

    """
    import xarray as xr

    TA = inputf['TA']
    ZH = inputf['ZH'].values
    dT = TA.values * 1
    dZ = ZH * 1
    # dZ = ZH(l) - ZH(l-1) [=] km
    dZ[:, 1:-1] = (ZH[:, 2:] - ZH[:, :-2]) / 1000
    # dT = TA(l) - TA(l-1) [=] K or C
    dT[:, 1:-1] = (dT[:, 2:] - dT[:, :-2])
    # Set lowest values to layer above them
    dT[:, 0] = dT[:, 1]
    dT[:, -1] = dT[:, -2]
    dZ[:, 0] = dZ[:, 1]
    dZ[:, -1] = dZ[:, -2]
    dTdZ = - dT / dZ
    # assuming upper trop layers are ~1km, using next layer
    # as a surrogate for higher levels within 2 km.
    dTdZ2 = dTdZ * 1
    dTdZ2[:, :-1] = - (dT[:, :-1] + dT[:, 1:]) / (dZ[:, :-1] + dZ[:, 1:])

    # Adding minimum 5km tropopause by not allowing a below min flag
    # below 5km.
    belowmin = np.ma.masked_where(
        ZH[:] < minval,
        np.ma.logical_and(
            dTdZ[:] < 2,
            dTdZ2[:] < 2
        )
    ).filled(False)
    istrop = np.cumsum(belowmin, axis=1) == 0

    return xr.DataArray(istrop, dims=TA.dims, coords=TA.coords, name='ISTROP')


def pressure_interp(PRESOUT, PRESIN, VAR, verbose=0, interp='numpy', **kwds):
    import xarray as xr
    if interp.lower() == 'numpy':
        def interp1d(data, x, xi, **kwds):
            return np.interp(xi, x, data, **kwds)
    else:
        def interp1d(data, x, xi, **kwds):
            from scipy import interpolate
            f = interpolate.interp1d(x, data, fill_value='extrapolate')
            return f(xi)

    X = PRESOUT.rename(LAY='NEW_LAY')
    XP = PRESIN.sortby('LAY')
    FP = VAR.sortby('LAY')
    interped = xr.apply_ufunc(
        interp1d,
        FP,
        XP,
        X,
        input_core_dims=[['LAY'], ['LAY'], ['NEW_LAY']],
        output_core_dims=[['NEW_LAY']],
        exclude_dims=set(("LAY",)),
        vectorize=True,
        kwargs=kwds
    )
    out = interped.rename(NEW_LAY='LAY').transpose(
        'TSTEP', 'LAY', 'ROW', 'COL'
    )
    return out
