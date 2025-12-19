.. _specfem_api_enumerations_connections:

``specfem::connections::type``
==============================

.. doxygenenum:: specfem::connections::type

``specfem::connections::to_string``
===================================

.. doxygenfunction:: specfem::connections::to_string(const specfem::connections::type& connection_type)

``specfem::connections::connection_mapping``
============================================

.. doxygenclass:: specfem::connections::connection_mapping
   :members:

``specfem::connections::to_string``
===================================

.. doxygenfunction:: specfem::connections::to_string

Dimension-Specific Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: specfem::connections::connection_mapping< specfem::dimension::type::dim2 >
    :members:

.. doxygenclass:: specfem::connections::connection_mapping< specfem::dimension::type::dim3 >
    :members:
