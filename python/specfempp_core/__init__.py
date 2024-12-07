from ._core import initialize, execute, finalize
import atexit

if initialize([]):
    atexit.register(finalize)

__all__ = ["initialize", "execute", "finalize"]
