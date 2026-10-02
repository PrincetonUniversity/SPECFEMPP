``specfem::assembly`` coordinate resolver
=========================================

Resolves generic source/receiver coordinates to mesh-space global coordinates.
The base handles Cartesian input (and dim2); projection modes derive from it.

``coordinate_resolver``
+++++++++++++++++++++++

.. doxygenclass:: specfem::assembly::coordinate_resolver
    :members:

``projecting_resolver``
+++++++++++++++++++++++

.. doxygenclass:: specfem::assembly::projecting_resolver
    :members:

``utm_resolver``
++++++++++++++++

.. doxygenclass:: specfem::assembly::utm_resolver
    :members:

``spherical_resolver``
++++++++++++++++++++++

.. doxygenclass:: specfem::assembly::spherical_resolver
    :members:

``make_coordinate_resolver``
++++++++++++++++++++++++++++

.. doxygenfunction:: specfem::assembly::make_coordinate_resolver
