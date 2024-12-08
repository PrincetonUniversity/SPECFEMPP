import atexit

from ._core import _initialize, _finalize
from .main import (
    set_par,
    set_default_par,
    load_par,
    load_default_par,
    get_par,
    get_default_par,
    execute,
)

if _initialize([]):
    atexit.register(_finalize)

__all__ = [
    "set_par",
    "set_default_par",
    "load_par",
    "load_default_par",
    "get_par",
    "get_default_par",
    "execute",
]
