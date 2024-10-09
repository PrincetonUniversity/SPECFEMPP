
.. _assembly_index:

Finite Element Assembly
-----------------------

.. doxygennamespace:: specfem::compute
    :desc-only:

.. doxygenstruct:: specfem::compute::assembly
    :members:

.. admonition:: Feature request
    :class: hint

    We need to define data access functions for the following data containers:

    1. Sources
    2. Receivers
    3. Coupled interfaces

    If you'd like to work on this, please see `issue tracker <https://github.com/PrincetonUniversity/SPECFEMPP/issues/110>_` for more details.

.. toctree::
    :maxdepth: 1

    mesh/mesh
    partial_derivatives/partial_derivatives
    properties/properties
    boundary/boundary
    fields/fields
    coupled_interfaces/coupled_interfaces
    sources/sources
    receivers/receivers
    kernels/kernels
