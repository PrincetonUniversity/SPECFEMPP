
.. _specfem_api_enumerations_wavefield:

``specfem::wavefield::simulation_field``
=======================================

.. doxygenenum:: specfem::wavefield::simulation_field

``specfem::wavefield::type``
============================

.. doxygenenum:: specfem::wavefield::type

``specfem::wavefield::wavefield``
=================================

.. doxygenclass:: specfem::wavefield::wavefield
   :members:

``specfem::wavefield::to_string``
=================================

.. doxygenfunction:: specfem::wavefield::to_string(const specfem::wavefield::type &wavefield_component)


Wavefield Specializations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: specfem::wavefield::wavefield< specfem::dimension::type::dim2, specfem::wavefield::type::displacement >
   :members:

.. doxygenclass:: specfem::wavefield::wavefield< specfem::dimension::type::dim2, specfem::wavefield::type::velocity >
   :members:

.. doxygenclass:: specfem::wavefield::wavefield< specfem::dimension::type::dim2, specfem::wavefield::type::acceleration >
   :members:

.. doxygenclass:: specfem::wavefield::wavefield< specfem::dimension::type::dim2, specfem::wavefield::type::pressure >
   :members:
