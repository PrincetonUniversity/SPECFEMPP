.. _algorithms_locate_point:

``specfem::algorithms::locate_point``
=====================================

2D Overloads
~~~~~~~~~~~~

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::point::global_coordinates<specfem::element::dimension_tag::dim2> &coordinates, const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh)

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::point::local_coordinates<specfem::element::dimension_tag::dim2> &coordinates, const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh)

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::kokkos::HostTeam::member_type &team_member, const specfem::point::local_coordinates<specfem::element::dimension_tag::dim2> &coordinates, const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh)

3D Overloads
~~~~~~~~~~~~

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::point::global_coordinates<specfem::element::dimension_tag::dim3> &coordinates, const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh)

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::point::local_coordinates<specfem::element::dimension_tag::dim3> &coordinates, const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh)

.. doxygenfunction:: specfem::algorithms::locate_point(const specfem::kokkos::HostTeam::member_type &team_member, const specfem::point::local_coordinates<specfem::element::dimension_tag::dim3> &coordinates, const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh)

The 3-D inverse mapping iterates on element anchors shifted by the target
point, so the residual stays element-sized in single precision even for
Earth-scale coordinates. Iteration stops on the reference-coordinate step. When
candidate distances tie, an element in which the point lies inside is
preferred. The returned location error is the exact back-projection distance.

Edge Location Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: specfem::algorithms::locate_point_on_edge(const specfem::point::global_coordinates<specfem::element::dimension_tag::dim2> &coordinates, const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh, const int &ispec, const specfem::mesh_entity::dim2::type &constraint)

.. doxygenfunction:: specfem::algorithms::locate_point_on_edge(const type_real &coordinate, const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh, const int &ispec, const specfem::mesh_entity::dim2::type &constraint)

Surface Projection
~~~~~~~~~~~~~~~~~~~

.. doxygenenum:: specfem::algorithms::projection

.. doxygenfunction:: specfem::algorithms::project_onto_surface
