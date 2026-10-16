``specfem::coordinate_systems::geocentric_coordinates``
-------------------------------------------------------

.. doxygenstruct:: specfem::coordinate_systems::geocentric_coordinates
    :members:

``geocentric_projection_config``
++++++++++++++++++++++++++++++++

.. doxygenstruct:: specfem::coordinate_systems::geocentric_projection_config
    :members:

``to`` (geocentric :math:`\leftrightarrow` Cartesian)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. doxygenfunction:: specfem::coordinate_systems::to< specfem::coordinate_systems::cartesian_coordinates, specfem::coordinate_systems::geocentric_coordinates >

.. doxygenfunction:: specfem::coordinate_systems::to< specfem::coordinate_systems::geocentric_coordinates, specfem::coordinate_systems::cartesian_coordinates >

``to`` (geographic :math:`\leftrightarrow` geocentric)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. doxygenfunction:: specfem::coordinate_systems::to< specfem::coordinate_systems::geocentric_coordinates, specfem::coordinate_systems::geographic_coordinates, specfem::coordinate_systems::geocentric_projection_config >

.. doxygenfunction:: specfem::coordinate_systems::to< specfem::coordinate_systems::geographic_coordinates, specfem::coordinate_systems::geocentric_coordinates, specfem::coordinate_systems::geocentric_projection_config >
