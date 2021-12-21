__all__ = [
    '__version__', 'open_dataset', 'constants', 'utils',
    'xr', 'pd', 'np', 'plt'
]


__version__ = "0.0.1"

from ._core import xr, pd, np, plt
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
