.. _specfem_boundary_conditions_stacey:

``specfem::boundary_conditions::stacey``
========================================

.. doxygenfunction:: specfem::boundary_conditions::impl_apply_boundary_conditions(const stacey_type &, const PointBoundaryType &boundary, const PointPropertyType &property, const PointFieldType &field, PointAccelerationType &acceleration)

.. doxygenfunction:: specfem::boundary_conditions::impl_compute_mass_matrix_terms(const stacey_type &, const type_real dt, const PointBoundaryType &boundary, const PointPropertyType &property, PointMassMatrixType &mass_matrix)

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: specfem::boundary_conditions::impl::impl_base_elastic_psv_traction
.. doxygenfunction:: specfem::boundary_conditions::impl::impl_base_elastic_sh_traction
.. doxygenfunction:: specfem::boundary_conditions::impl::impl_base_elastic_psv_t_traction
.. doxygenfunction:: specfem::boundary_conditions::impl::impl_enforce_traction
