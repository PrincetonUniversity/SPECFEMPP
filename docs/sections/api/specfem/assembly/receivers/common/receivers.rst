.. _assembly_receivers_common:

``specfem::assembly::receivers`` (Template)
============================================

.. doxygenstruct:: specfem::assembly::receivers
   :members:

The main template class for assembly-level receiver management in spectral element
simulations. This template provides a unified interface for managing seismic receivers
across different spatial dimensions, with specializations for 2D and 3D implementations.

Key capabilities:

* Lagrange interpolation for accurate field sampling at receiver locations
* Efficient Kokkos-based data structures for host/device computations
* Support for multiple seismogram types (displacement, velocity, acceleration)
* Coordinate transformations for proper seismogram orientation
* Integration with spectral element mesh structures
