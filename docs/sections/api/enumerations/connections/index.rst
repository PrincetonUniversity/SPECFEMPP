.. _specfem_api_enumerations_connections:

``specfem::element_connections::type``
==============================

.. doxygenenum:: specfem::element_connections::type

``specfem::element_connections::to_string``
===================================

.. doxygenfunction:: specfem::element_connections::to_string(const specfem::element_connections::type& connection_type)

``specfem::element_connections::connection_mapping``
============================================

.. doxygenclass:: specfem::element_connections::connection_mapping
   :members:

``specfem::element_connections::to_string``
===================================

.. doxygenfunction:: specfem::element_connections::to_string

Dimension-Specific Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: specfem::element_connections::connection_mapping< specfem::element::dimension_tag::dim2 >
    :members:

.. doxygenclass:: specfem::element_connections::connection_mapping< specfem::element::dimension_tag::dim3 >
    :members:
