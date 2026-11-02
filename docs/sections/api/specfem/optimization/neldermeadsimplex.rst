Nelder-Mead Simplex Optimization Algorithm
==========================================


.. doxygenstruct:: specfem::optimization::NelderMeadOptions
    :members:


.. doxygenfunction:: specfem::optimization::optimize(NelderMeadSimplex, Func &&objective, NelderMeadOptions<N, ValueType, Globe> options)

GLOBE compatibility
-------------------

Use ``NelderMeadOptions<N, double, true>`` with ``NelderMeadSimplex{}``
for SPECFEM3D_GLOBE's attenuation fit. This selects double precision,
Fortran-rounded defaults, exact-zero initialization, heap-sort tie ordering,
GLOBE trial-point expressions and inclusive convergence tolerances.
The initial simplex counts as iteration one. Both budgets default to
``200*N``; a started iteration completes even if it exceeds the evaluation
budget.

The default ``NelderMeadOptions<N>`` retains the standard solver behavior.

.. doxygenfunction:: specfem::optimization::globe_heap_sort
