Coordinate conversions
======================

Conversions are const members of the concrete coordinate types. Include
``specfem/coordinate_systems.hpp`` and call the member on the source coordinate:

.. code-block:: cpp

   const specfem::coordinate_systems::geographic_coordinates geographic{
       2.6741959317615298, 51.561449479910003, 100.0 };
   const specfem::coordinate_systems::utm_projection_config utm{ 31 };
   const auto cartesian = geographic.to_cartesian(utm);
   const auto recovered = cartesian.to_geographic(utm);

   const specfem::coordinate_systems::geocentric_projection_config globe{
       6371000.0 };
   const auto geocentric = geographic.to_geocentric(globe);
   const auto globe_cartesian = geocentric.to_cartesian();
   const auto spherical = globe_cartesian.to_geocentric();
   const auto globe_geographic = spherical.to_geographic(globe);

Supported conversions
---------------------

* ``geocentric.to_cartesian()`` returns 3D Cartesian coordinates.
* ``cartesian3.to_geocentric()`` returns geocentric coordinates.
* ``geographic.to_geocentric(geocentric_config)`` returns geocentric coordinates.
* ``geocentric.to_geographic(geocentric_config)`` returns geographic coordinates.
* ``geographic.to_cartesian(utm_config)`` returns 3D Cartesian coordinates.
* ``cartesian3.to_geographic(utm_config)`` returns geographic coordinates.

Projection configuration must be supplied explicitly for conversions involving
geographic coordinates. These conversions are not available on the abstract
``coordinates`` base or on 2D Cartesian coordinates.

Cartesian components must match the requested conversion: planet-centered XYZ
for ``to_geocentric()``, or UTM easting/northing for ``to_geographic(utm_config)``.
The optional Cartesian ``origin`` is not applied by these methods.

These members perform coordinate transformations only. Resolution to mesh-space
global coordinates, including depth placement against the free surface, remains
in ``specfem::assembly::to_cartesian`` and its reverse assembly conversions.
