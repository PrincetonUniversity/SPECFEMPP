.. _assembly_receivers_dim2:

2D Receiver Specialization
===========================

2D receivers specialization for ``specfem::assembly::receivers<specfem::dimension::type::dim2>``.

This specialization handles seismic receivers in 2D spectral element simulations with
features specific to two-dimensional problems:

* Support for multiple medium types (elastic_psv, elastic_sh, acoustic, poroelastic)
* Angle-based coordinate rotation using sine/cosine transformations
* 2-component seismogram recording (horizontal and vertical components)
* Efficient memory layout optimized for 2D computations

Class Documentation
-------------------

.. doxygenstruct:: specfem::assembly::receivers< specfem::dimension::type::dim2 >
   :members:

Data Access Functions
---------------------

.. doxygengroup:: ComputeReceiversDataAccess2D
   :members:
