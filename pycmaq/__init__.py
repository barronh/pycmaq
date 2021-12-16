__all__ = ['__version__', 'open_dataset', 'xr', 'pd', 'np', 'plt']


__version__ = "0.0.1"

from ._core import xr, pd, np, plt


def open_dataset(*args, **kwds):
    from copy import deepcopy
    kwds = deepcopy(kwds)
    kwds.setdefault('backend_kwargs', dict(mode='rs'))
    return xr.open_dataset(*args, **kwds)
