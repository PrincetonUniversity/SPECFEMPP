Fminsearch (SPECFEM3D_GLOBE simplex)
====================================

.. doxygenstruct:: specfem::optimization::Fminsearch

.. doxygenstruct:: specfem::optimization::FminsearchOptions
    :members:

.. doxygenfunction:: specfem::optimization::optimize(Fminsearch tag, Func &&objective, FminsearchOptions<N> options)

.. doxygenfunction:: specfem::optimization::fminsearch_heap_sort

Both budgets default to ``200*N``. The initial simplex counts as iteration one
and costs ``N+1`` evaluations. Budgets are checked at iteration boundaries, so
a started contraction or shrink always completes. Convergence includes
equality with the tolerances. ``function_evaluations`` reports the number of
objective calls. ``NelderMeadSimplex`` is unchanged and remains the
general-purpose solver used by 2D attenuation.
