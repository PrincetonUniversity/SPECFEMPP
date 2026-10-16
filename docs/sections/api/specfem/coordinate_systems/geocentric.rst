``specfem::coordinate_systems::geocentric_coordinates``
-------------------------------------------------------

.. doxygenstruct:: specfem::coordinate_systems::geocentric_coordinates
    :members:

``geocentric_projection_config``
++++++++++++++++++++++++++++++++

.. doxygenstruct:: specfem::coordinate_systems::geocentric_projection_config
    :members:

Geocentric and Cartesian conversions
++++++++++++++++++++++++++++++++++++

Call ``geocentric.to_cartesian()`` or ``cartesian3.to_geocentric()`` on a const
concrete coordinate. These conversions do not require projection configuration.

Geographic and geocentric conversions
++++++++++++++++++++++++++++++++++++

Call ``geographic.to_geocentric(config)`` or
``geocentric.to_geographic(config)`` with a
``geocentric_projection_config`` specifying the planet radius.
See :doc:`to` for examples.
