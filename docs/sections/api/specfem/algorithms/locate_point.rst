.. _algorithms_locate_point:

``specfem::algorithms::locate_point``
=====================================

2D Overloads
~~~~~~~~~~~~

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::point::global_coordinates<specfem::dimension::type::dim2> &coordinates, const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh)

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::point::local_coordinates<specfem::dimension::type::dim2> &coordinates, const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh)

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::kokkos::HostTeam::member_type &team_member, const specfem::point::local_coordinates<specfem::dimension::type::dim2> &coordinates, const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh)

3D Overloads
~~~~~~~~~~~~

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::point::global_coordinates<specfem::dimension::type::dim3> &coordinates, const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh)

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::point::local_coordinates<specfem::dimension::type::dim3> &coordinates, const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh)

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::kokkos::HostTeam::member_type &team_member, const specfem::point::local_coordinates<specfem::dimension::type::dim3> &coordinates, const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh)

Edge Location Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: specfem::algorithms::locate_point_on_edge(const specfem::point::global_coordinates<specfem::dimension::type::dim2> &coordinates, const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh, const int &ispec, const specfem::mesh_entity::dim2::type &constraint)

.. doxygenfunction:: specfem::algorithms::locate_point_on_edge(const type_real &coordinate, const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh, const int &ispec, const specfem::mesh_entity::dim2::type &constraint)
