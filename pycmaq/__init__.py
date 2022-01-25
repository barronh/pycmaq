__all__ = [
    '__version__', 'CmaqAccessor', 'open_dataset', 'open_griddesc', 'constants',
    'utils', 'xr', 'pd', 'np', 'plt'
]


__version__ = "0.0.1"

from ._core import CmaqAccessor, xr, pd, np, plt
from . import constants
from . import utils


def open_dataset(*args, **kwds):
    """
    Thin wrapper around xarray.open_dataset:
    * adds a default backend_kwargs with mode='rs' before opening, and
    * applies cmaq.set_coords method after opening.
    """
    from copy import deepcopy
    kwds = deepcopy(kwds)
    kwds.setdefault('backend_kwargs', dict(mode='rs'))
    outf = xr.open_dataset(*args, **kwds)
    outf.cmaq.set_coords()
    return outf


def open_griddesc(gdpath, GDNAM, **kwds):
    """
    Simple wrapper around around open_dataset with pseudonetcdf
    and backend_kwargs specified.

    If gdpath is None, use the default GRIDDESC. For details, use gdpath=None,
    help=True.
    """
    if gdpath is None:
        from . import default_griddesc
        return default_griddesc.open_default_griddesc(GDNAM=GDNAM, **kwds)
    kwds = kwds.copy()
    kwds['format'] = 'griddesc'
    kwds['GDNAM'] = GDNAM
    outf = open_dataset(
        gdpath, engine='pseudonetcdf',
        backend_kwargs=kwds
    )
    return outf


def from_dataframe(df, **ioapi_kw):
    """
    Thin wrapper around xr.Dataset.cmaq.from_dataframe for convenience
    as pycmaq.from_dataframe. Requires complete dataframe.

    ROW and COL indices should range from 0.5 ... N - 0.5
    LAY indices should match (VGLVLS[k] + VGLVLS[k + 1]) / 2
    TSTEP should have np.datetime64 or pd.Time
    """
    return xr.Dataset.cmaq.from_dataframe(df, **ioapi_kw)


def from_dataframe_incomplete(df, fill_value=None, **ioapi_kw):
    """
    Thin wrapper around xr.Dataset.cmaq.from_dataframe_incomplete for
    convenience as pycmaq.from_dataframe. Missing values will be filled with
    nan before creating the Dataset.
    """
    return xr.Dataset.cmaq.from_dataframe_incomplete(
        df, fill_value=fill_value, **ioapi_kw
    )
