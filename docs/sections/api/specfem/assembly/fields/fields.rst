
.. _assembly_fields:

``specfem::assembly::fields``
=============================

.. doxygenstruct:: specfem::assembly::fields
    :members:

``specfem::assembly::simulation_field``
=======================================

.. doxygenstruct:: specfem::assembly::simulation_field
    :members:

Dimension-Specific Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: specfem::assembly::simulation_field< specfem::dimension::type::dim2, SimulationWavefieldType >
    :members:

.. doxygenstruct:: specfem::assembly::simulation_field< specfem::dimension::type::dim3, SimulationWavefieldType >
    :members:

Data Access Functions
^^^^^^^^^^^^^^^^^^^^^

.. doxygengroup:: FieldsDataAccess
    :content-only:

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   field_impl
   base_field
