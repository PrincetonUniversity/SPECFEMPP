``specfem::assembly::to``
=========================

Converts a source/receiver input coordinate (Cartesian, geographic, or
geocentric) to the target coordinate system using the mesh context. The
projection is selected statically from the mesh model (UTM for a regional
Cartesian3D mesh, the geographic → geocentric → Cartesian composition for a
Globe3D mesh, pass-through otherwise), so callers never pass a projection
configuration. A Cartesian target returns the mesh-space
``specfem::point::global_coordinates`` consumed by
:doc:`locate_point <../../algorithms/locate_point/locate_point>`; other targets
return their own coordinate type.

2D conversion
-------------

.. doxygenfunction:: specfem::assembly::to(const specfem::coordinate_systems::coordinates< specfem::element::dimension_tag::dim2 > &input, const specfem::assembly::mesh< specfem::element::dimension_tag::dim2 > &mesh, const specfem::mesh::mesh< ModelTag > &raw_mesh)

3D conversion
-------------

.. doxygenfunction:: specfem::assembly::to(const specfem::coordinate_systems::coordinates< specfem::element::dimension_tag::dim3 > &input, const specfem::assembly::mesh< specfem::element::dimension_tag::dim3 > &mesh, const specfem::mesh::mesh< ModelTag > &raw_mesh)
