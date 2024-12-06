"""
SPECFEM++ Python
-----------------------

.. currentmodule:: specfempp

.. autosummary::
    :toctree: _generate

    _run
    subtract
"""

def _run(argv: list[str]) -> int:
    """
    Run main specfem workflow

    Args:
        argv: list of command line arguments for initializing MPI and Kokkos
    """
