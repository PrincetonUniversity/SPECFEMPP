
.. _receivers_api:

``specfem::receivers``
======================

.. doxygennamespace:: specfem::receivers
    :desc-only:

3-D receivers given in geographic coordinates are placed through the mesh projection, and their seismograms are written in the local East/North/Up frame. See :doc:`../coordinate_systems/spherical_basis` for ``specfem::coordinate_systems::GeographicBasis`` and :doc:`../assembly/coordinate_conversion/coordinate_conversion` for ``specfem::assembly::local_geographic_basis``.


``specfem::receivers::receiver``
++++++++++++++++++++++++++++++++

.. doxygenclass:: specfem::receivers::receiver
   :members:
   :undoc-members:
   :private-members:


2D Specialization
-----------------

.. doxygenclass:: specfem::receivers::receiver< specfem::element::dimension_tag::dim2 >
   :members:
   :undoc-members:
   :private-members:


3D Specialization
-----------------

.. doxygenclass:: specfem::receivers::receiver< specfem::element::dimension_tag::dim3 >
   :members:
   :undoc-members:
   :private-members:
