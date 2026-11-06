Coordinate conversion
======================

Converts a source/receiver input coordinate (Cartesian, geographic, or
geocentric) into mesh space, and converts a mesh-space Cartesian point back into
another coordinate system. The projection is selected by the raw-mesh type
(pass-through for a plain Cartesian mesh, UTM for a regional Cartesian3D mesh,
the geographic → geocentric → Cartesian composition for a Globe3D mesh), so
callers never pass a projection configuration. ``to_cartesian`` returns the
mesh-space ``specfem::point::global_coordinates`` consumed by
:doc:`locate_point <../../algorithms/locate_point/locate_point>`; the reverse
conversions return their own coordinate type.

Placement into mesh space (``to_cartesian``)
--------------------------------------------

.. doxygenfunction:: specfem::assembly::to_cartesian(const specfem::coordinate_systems::coordinates< specfem::element::dimension_tag::dim2 > &input, const specfem::assembly::mesh< specfem::element::dimension_tag::dim2 > &mesh, const specfem::mesh::cartesian2d_mesh &raw_mesh)

.. doxygenfunction:: specfem::assembly::to_cartesian(const specfem::coordinate_systems::coordinates< specfem::element::dimension_tag::dim3 > &input, const specfem::assembly::mesh< specfem::element::dimension_tag::dim3 > &mesh, const specfem::mesh::cartesian3d_mesh &raw_mesh)

.. doxygenfunction:: specfem::assembly::to_cartesian(const specfem::coordinate_systems::coordinates< specfem::element::dimension_tag::dim3 > &input, const specfem::assembly::mesh< specfem::element::dimension_tag::dim3 > &mesh, const specfem::mesh::globe3d_mesh &raw_mesh)

Reverse conversions (``to_geographic`` / ``to_geocentric``)
-----------------------------------------------------------

.. doxygenfunction:: specfem::assembly::to_geographic(const specfem::coordinate_systems::coordinates< specfem::element::dimension_tag::dim3 > &input, const specfem::assembly::mesh< specfem::element::dimension_tag::dim3 > &mesh, const specfem::mesh::cartesian3d_mesh &raw_mesh)

.. doxygenfunction:: specfem::assembly::to_geographic(const specfem::coordinate_systems::coordinates< specfem::element::dimension_tag::dim3 > &input, const specfem::assembly::mesh< specfem::element::dimension_tag::dim3 > &mesh, const specfem::mesh::globe3d_mesh &raw_mesh)

.. doxygenfunction:: specfem::assembly::to_geocentric(const specfem::coordinate_systems::coordinates< specfem::element::dimension_tag::dim3 > &input, const specfem::assembly::mesh< specfem::element::dimension_tag::dim3 > &mesh, const specfem::mesh::globe3d_mesh &raw_mesh)
