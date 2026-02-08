.. _assembly_sources_index:

``specfem::assembly::sources``
==============================

.. doxygenstruct:: specfem::assembly::sources
    :members:

Dimension-Specific Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: specfem::assembly::sources< specfem::element::dimension_tag::dim2 >
    :members:

.. doxygenstruct:: specfem::assembly::sources< specfem::element::dimension_tag::dim3 >
    :members:

Data Access Functions
^^^^^^^^^^^^^^^^^^^^^

.. doxygengroup:: SourceDataAccess
    :content-only:

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   locate_sources
   source_medium
   sort_sources_per_medium
