.. _kokkos_abstractions:

Views and Execution Policies
============================

Specfem abstractions for Kokkos interfaces. The goal is never to use Kokkos views or execution policies directly. Ideally we would want to define them in kokkos_abstractions.h file under the specfem namespace.

This keeps definition of datatypes/execution-policies consistent throughout the whole project.

.. doxygennamespace:: specfem::kokkos
   :members:
   :undoc-members:
