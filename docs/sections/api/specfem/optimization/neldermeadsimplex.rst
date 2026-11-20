Nelder-Mead Simplex Optimization Algorithm
==========================================


.. doxygenstruct:: specfem::optimization::NelderMeadOptions
    :members:


.. doxygenfunction:: specfem::optimization::optimize(NelderMeadSimplex, Func &&objective, NelderMeadOptions<N, ValueType> options)

Semantics
---------

The solver follows ``fminsearch`` (Lagarias et al. 1998), the variant used by
the SPECFEM Fortran codes. The initial simplex counts as iteration one, both
budgets default to ``200*N``, a started iteration completes even if it exceeds
the evaluation budget, and both convergence tolerances are inclusive. Vertices
are ordered with a heap sort, so ties are ordered deterministically and as in
the Fortran codes.

Callers that must reproduce a Fortran run bit for bit set its constants
through the options; the SPECFEM3D_GLOBE attenuation fit, for example, uses
double precision with single-precision ``tol_f``, ``tol_x``,
``initial_step`` and ``zero_step``.

.. doxygenfunction:: specfem::optimization_impl::heap_sort
