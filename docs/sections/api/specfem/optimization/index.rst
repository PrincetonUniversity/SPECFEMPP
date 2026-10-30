``specfem::optimization``
=========================

.. doxygennamespace:: specfem::optimization
    :desc-only:

The result scalar type defaults to ``type_real`` and can be selected explicitly
for setup calculations that require double precision.

.. doxygenstruct:: specfem::optimization::OptimizationResult


.. doxygenfunction:: specfem::optimization::optimize(AlgorithmTag tag, Func &&objective, Options options)


.. doxygenfunction:: specfem::optimization::optimize(AlgorithmTag tag, Func &&objective, GradFunc &&gradient, Options options)




Algorithms
----------

.. doxygenstruct:: specfem::optimization::NelderMeadSimplex

.. doxygenstruct:: specfem::optimization::Fminsearch

.. doxygenstruct:: specfem::optimization::SteepestDescent


.. toctree::
    :maxdepth: 1

    neldermeadsimplex
    fminsearch
    steepestdescent
