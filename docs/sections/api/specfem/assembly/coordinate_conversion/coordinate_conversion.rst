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

.. doxygenfunction:: specfem::assembly::to_cartesian(const specfem::coordinate_systems::coordinates< specfem::element::dimension_tag::dim2 > &input, const specfem::assembly::mesh< specfem::element::dimension_tag::dim2 > &mesh)

.. doxygenfunction:: specfem::assembly::to_cartesian(const specfem::coordinate_systems::coordinates< specfem::element::dimension_tag::dim3 > &input, const specfem::assembly::mesh< specfem::element::dimension_tag::dim3 > &mesh)

Reverse conversions (``to_geographic`` / ``to_geocentric``)
-----------------------------------------------------------

.. doxygenfunction:: specfem::assembly::to_geographic

.. doxygenfunction:: specfem::assembly::to_geocentric

Local frame at a mesh point
---------------------------

``local_frame_rotation`` is the single owner of which local frame a mesh-space
point has: it returns the :doc:`frame-tagged rotation
<../../coordinate_systems/basis_rotation>` from the local geocentric
``(theta, phi, r)`` axes into mesh ``(x, y, z)`` components. On a globe mesh
the frame rotates with the point's colatitude and longitude; on a UTM or plain
Cartesian mesh it is the fixed ``flat_mesh_rotation`` (x = East, y = North,
z = Up). Sources rotate their moment tensor or force with it;
``local_east_north_up_basis`` turns it into the East/North/Up rows stored per
receiver (see :doc:`east_north_up <../../coordinate_systems/east_north_up>` for
the relabel), which is the identity on a flat mesh.

.. doxygentypedef:: specfem::assembly::LocalFrameRotation

.. doxygenfunction:: specfem::assembly::flat_mesh_rotation

.. doxygenfunction:: specfem::assembly::local_frame_rotation

.. doxygenfunction:: specfem::assembly::local_east_north_up_basis
