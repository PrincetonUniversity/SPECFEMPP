.. _kokkos_abstractions:

SPECFEM abstractions for Kokkos interfaces
===========================================

Specfem abstractions for Kokkos interfaces. The goal is never to use Kokkos views or execution policies directly. Ideally we would want to define them in kokkos_abstractions.h file under the specfem namespace.

This keeps definition of datatypes/execution-policies consistent throughout the whole project.

.. doxygenfile:: kokkos_abstractions.h
   :project: SPECFEM KOKKOS IMPLEMENTATION
