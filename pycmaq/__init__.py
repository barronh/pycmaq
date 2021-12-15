__all__ = ['__version__', 'open_dataset']


__version__ = "0.0.1"

from ._core import xr

open_dataset = xr.open_dataset
