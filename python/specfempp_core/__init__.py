import atexit

from ._core import _initialize, _finalize, get_seismograms
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


def run_forward():
    from obspy import Stream, Trace
    import numpy as np

    execute()
    m = get_seismograms()
    trs = []

    for sta in m:
        if sta == "__time__":
            continue
        n, s, c, _ = sta.split(".")
        trs.append(
            Trace(
                data=np.array(m[sta]), header={"station": s, "network": n, "channel": c}
            )
        )

    return Stream(traces=trs)


__all__ = [
    "set_par",
    "set_default_par",
    "load_par",
    "load_default_par",
    "get_par",
    "get_default_par",
    "execute",
    "get_seismograms",
    "run_forward",
]
