Steepest Descent Optimization Algorithm
=======================================

.. doxygenstruct:: specfem::optimization::SteepestDescentOptions
    :members:

.. doxygenfunction:: specfem::optimization::optimize(SteepestDescent, Func &&objective, SteepestDescentOptions<N> options)

.. doxygenfunction:: specfem::optimization::optimize(SteepestDescent, Func &&objective, GradFunc &&gradient, SteepestDescentOptions<N> options)
