Moment-tensor rotation
----------------------

Rotation of a seismic moment tensor from the spherical :math:`(r, \theta, \phi)`
(CMTSOLUTION) basis to the global Cartesian basis, plus the flat
East-North-Up relabel used on regional meshes. The rotation consumes the local
frame from :doc:`../coordinate_systems/spherical_basis`.

.. doxygenstruct:: specfem::sources::SphericalMomentTensor
    :members:

.. doxygenstruct:: specfem::sources::CartesianMomentTensor
    :members:

.. doxygenfunction:: specfem::sources::rotate_moment_tensor_spherical_to_cartesian

.. doxygenfunction:: specfem::sources::spherical_to_enu
