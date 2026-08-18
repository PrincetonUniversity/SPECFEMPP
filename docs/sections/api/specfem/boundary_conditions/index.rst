
.. _boundary_conditions_api:

``specfem::boundary_conditions``
================================


.. doxygennamespace:: specfem::boundary_conditions
    :desc-only:


``specfem::boundary_conditions::apply_boundary_conditions``
-----------------------------------------------------------

.. doxygenfunction:: specfem::boundary_conditions::apply_boundary_conditions(const PointBoundaryType &boundary, PointAccelerationType &acceleration)

.. doxygenfunction:: specfem::boundary_conditions::apply_boundary_conditions(const PointBoundaryType &boundary, const PointPropertyType &property, const PointVelocityType &field, PointAccelerationType &acceleration)

``specfem::boundary_conditions::compute_mass_matrix_terms``
-----------------------------------------------------------

.. doxygenfunction:: specfem::boundary_conditions::compute_mass_matrix_terms(const type_real dt, const PointBoundaryType &boundary, const PointPropertyType &property, PointMassMatrixType &mass_matrix)

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   none
   stacey
   dirichlet
   composite_stacey_dirichlet
