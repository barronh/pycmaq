__all__ = ['__version__', 'open_dataset', 'xr', 'pd', 'np', 'plt']


__version__ = "0.0.1"

from ._core import xr, pd, np, plt

open_dataset = xr.open_dataset
