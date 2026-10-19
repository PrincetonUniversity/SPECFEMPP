Force rotation
--------------

Rotation of a collocated force vector from the local East-North-Up (geographic)
frame to the global Cartesian frame, plus the flat regional relabel (the
identity for a vector). The rotation consumes the local frame from
:doc:`../coordinate_systems/spherical_basis`.

.. doxygenstruct:: specfem::sources::GeographicForce
    :members:

.. doxygenstruct:: specfem::sources::CartesianForce
    :members:

.. doxygenfunction:: specfem::sources::rotate_force_geographic_to_cartesian

.. doxygenfunction:: specfem::sources::geographic_to_enu(const specfem::sources::GeographicForce&)
