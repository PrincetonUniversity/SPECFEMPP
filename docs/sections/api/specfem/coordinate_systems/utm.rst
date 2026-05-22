UTM projection
--------------

``utm_projection_config``
+++++++++++++++++++++++++

.. doxygenstruct:: specfem::coordinate_systems::utm_projection_config
    :members:

``transform`` (UTM specializations)
+++++++++++++++++++++++++++++++++++

.. doxygenfunction:: specfem::coordinate_systems::transform< specfem::coordinate_systems::cartesian_coordinates, specfem::coordinate_systems::geographic_coordinates, specfem::coordinate_systems::utm_projection_config >

.. doxygenfunction:: specfem::coordinate_systems::transform< specfem::coordinate_systems::geographic_coordinates, specfem::coordinate_systems::cartesian_coordinates, specfem::coordinate_systems::utm_projection_config >
