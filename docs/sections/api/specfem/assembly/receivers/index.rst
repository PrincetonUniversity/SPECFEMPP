.. _assembly_receivers_index:

``specfem::assembly::receivers``
================================

The assembly receivers module manages seismic receiver information within assembled
finite element meshes for spectral element simulations. Receivers handle seismogram
recording with various output types and coordinate transformations for proper
seismogram orientation based on receiver geometry.

Common Templates
----------------

.. toctree::
   :maxdepth: 1

   common/receivers
   common/receiver_iterator

Dimension-Specific Implementations
----------------------------------

.. toctree::
   :maxdepth: 1

   dim2/index
   dim3/index
