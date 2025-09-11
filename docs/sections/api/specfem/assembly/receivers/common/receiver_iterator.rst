.. _assembly_receivers_iterator:

Receiver Iterator Classes
=========================

Station Iterator
----------------

.. doxygenclass:: specfem::assembly::receivers_impl::StationIterator
   :members:

Station Info
------------

.. doxygenstruct:: specfem::assembly::receivers_impl::StationInfo
   :members:

Seismogram Type Iterator
------------------------

.. doxygenclass:: specfem::assembly::receivers_impl::SeismogramTypeIterator
   :members:

Seismogram Iterator Template
----------------------------

.. doxygenclass:: specfem::assembly::receivers_impl::SeismogramIterator
   :members:

These classes provide iterator-based access to receiver station metadata and seismogram
data. The StationIterator manages station names and network information, while the
SeismogramIterator handles time-series data access with dimension-specific specializations
for coordinate transformations and component handling.
