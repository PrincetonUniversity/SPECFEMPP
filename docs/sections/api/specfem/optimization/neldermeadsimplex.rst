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
are ordered with a stable sort, as MATLAB's ``fminsearch`` does, so tied
vertices keep their order.

Callers set the Fortran codes' constants through the options; the
attenuation Q fit, for example, runs in double precision with the
single-precision ``tol_f``, ``tol_x``, ``initial_step`` and ``zero_step`` of
the Fortran ``fminsearch``.
