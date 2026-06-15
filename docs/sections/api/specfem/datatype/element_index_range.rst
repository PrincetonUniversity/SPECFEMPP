.. _specfem_api_datatype_element_index_range:

``specfem::datatype::ElementIndexRange``
========================================

Zero-allocation contiguous range container for element indices.

.. doxygenclass:: specfem::datatype::ElementIndexRange
    :members:

Free Functions
^^^^^^^^^^^^^^

.. doxygenfunction:: specfem::datatype::subview(const ElementIndexRange &range, Kokkos::pair<int, int> slice)
