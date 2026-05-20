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

Projections
-----------

*  :doc:`utm`: UTM (Universal Transverse Mercator) forward and inverse projection.

.. toctree::
    :maxdepth: 1

    geographic
    cartesian
    geocentric
    utm
