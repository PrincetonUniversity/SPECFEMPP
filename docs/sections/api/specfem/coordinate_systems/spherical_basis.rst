``specfem::coordinate_systems::local_spherical_basis``
------------------------------------------------------

The local spherical orthonormal basis :math:`(\mathbf{e}_r, \mathbf{e}_\theta,
\mathbf{e}_\phi)` at a point on the sphere, expressed in global Cartesian
components. This is the single owner of the spherical-basis trigonometry: the
source moment-tensor rotation (rank-2) and the receiver seismogram-component
rotation (rank-1) each build their rotation matrix from it.

.. doxygenstruct:: specfem::coordinate_systems::SphericalBasis
    :members:

.. doxygenfunction:: specfem::coordinate_systems::local_spherical_basis

``specfem::coordinate_systems::local_geographic_basis``
-------------------------------------------------------

The local geographic (East, North, Up) basis, built from the spherical basis
with East :math:`= \mathbf{e}_\phi`, North :math:`= -\mathbf{e}_\theta` and
Up :math:`= \mathbf{e}_r`. It is right-handed, and it is the single owner of
this sign convention: force sources rotate from it into the mesh and receivers
rotate seismograms from the mesh into it.

.. doxygenstruct:: specfem::coordinate_systems::GeographicBasis
    :members:

.. doxygenfunction:: specfem::coordinate_systems::local_geographic_basis
