UTM projection
--------------

``utm_projection_config``
+++++++++++++++++++++++++

.. doxygenstruct:: specfem::coordinate_systems::utm_projection_config
    :members:

UTM conversions
+++++++++++++++

Call ``geographic.to_cartesian(config)`` for the forward projection or
``cartesian3.to_geographic(config)`` for the inverse projection. Both are const
members of the concrete coordinate types and require a
``utm_projection_config``. See :doc:`to` for examples.
