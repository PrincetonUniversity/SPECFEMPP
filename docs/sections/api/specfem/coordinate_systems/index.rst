.. _specfem_api_coordinate_systems_index:

``specfem::coordinate_systems``
================================

.. doxygennamespace:: specfem::coordinate_systems
    :desc-only:

The ``specfem::coordinate_systems`` namespace provides coordinate types and
map projections for geodetic I/O. Include via
``#include "specfem/coordinate_systems.hpp"``.

Coordinate types
-----------------

*  :doc:`geographic_coordinates <geographic>`: Longitude/latitude in degrees, depth in meters.
*  :doc:`cartesian_coordinates <cartesian>`: x/y/z in meters (easting/northing/depth for UTM).
*  :doc:`geocentric_coordinates <geocentric>`: Spherical :math:`(r, \theta, \phi)` coordinates.

Transforms
----------

*  :doc:`basis_rotation`: Device-callable orthonormal basis rotations for
   vectors and tensors between global Cartesian and local radial axes.
*  :doc:`to`: Generic ``to<Target>(source[, config])`` interface for
   converting between coordinate systems. Resolution to mesh-space global
   coordinates (which needs the mesh and projection configuration) is handled by
   ``specfem::assembly::to_cartesian`` (and the reverse ``to_geographic`` /
   ``to_geocentric``) in the assembly layer.

Projections
-----------

*  :doc:`utm`: UTM (Universal Transverse Mercator) forward and inverse projection.

Local basis
-----------

*  :doc:`spherical_basis`: Local spherical orthonormal basis
   :math:`(\mathbf{e}_r, \mathbf{e}_\theta, \mathbf{e}_\phi)` at a point, used
   to build rotation matrices for source and receiver quantities.

.. toctree::
    :maxdepth: 1

    geographic
    cartesian
    geocentric
    basis_rotation
    to
    utm
    spherical_basis
