``specfem::optimization``
=========================

.. doxygennamespace:: specfem::optimization
    :desc-only:

.. doxygenstruct:: specfem::optimization::OptimizationResult


.. doxygenfunction:: specfem::optimization::optimize(AlgorithmTag tag, Func &&objective, Options options)


.. doxygenfunction:: specfem::optimization::optimize(AlgorithmTag tag, Func &&objective, GradFunc &&gradient, Options options)




Algorithms
----------

.. doxygenstruct:: specfem::optimization::NelderMeadSimplex

.. doxygenstruct:: specfem::optimization::SteepestDescent


.. toctree::
    :maxdepth: 1

    neldermeadsimplex
    steepestdescent
