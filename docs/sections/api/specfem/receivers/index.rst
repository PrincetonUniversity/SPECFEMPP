
.. _receivers_api:

``specfem::receivers``
======================

.. doxygennamespace:: specfem::receivers
    :desc-only:

3-D receivers given in geographic coordinates are placed through the mesh projection, and their seismograms are written in the local East/North/Up frame. The frame comes from :doc:`../coordinate_systems/basis_rotation` (the local ``(theta, phi, r)`` axes at the receiver) relabelled through :doc:`../coordinate_systems/east_north_up`; see :doc:`../assembly/coordinate_conversion/coordinate_conversion` for ``specfem::assembly::local_frame_rotation`` and ``specfem::assembly::local_east_north_up_basis``, which build the per-receiver rotation rows.


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
