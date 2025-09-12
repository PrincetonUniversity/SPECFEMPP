.. _assembly_receivers_dim3:

3D Receiver Specialization
===========================

3D receivers specialization for ``specfem::assembly::receivers<specfem::dimension::type::dim3>``.

This specialization handles seismic receivers in 3D spectral element simulations with
features specific to three-dimensional problems:

* Currently supports elastic medium type
* Full 3x3 rotation matrix transformations for arbitrary receiver orientations
* 3-component seismogram recording (X, Y, Z or North, East, Up components)
* Advanced rotation capabilities for complex receiver geometries

Class Documentation
-------------------

.. doxygenstruct:: specfem::assembly::receivers< specfem::dimension::type::dim3 >
   :members:

Data Access Functions
---------------------

.. doxygengroup:: ComputeReceiversDataAccess3D
   :members:
