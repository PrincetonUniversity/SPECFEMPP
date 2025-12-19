.. _specfem_api_enumerations_medium:

``specfem::element::medium_tag``
================================

.. doxygenenum:: specfem::element::medium_tag

``specfem::element::property_tag``
==================================

.. doxygenenum:: specfem::element::property_tag

``specfem::element::boundary_tag``
==================================

.. doxygenenum:: specfem::element::boundary_tag

``specfem::element::to_string``
================================

.. doxygenfunction:: specfem::element::to_string(const medium_tag &medium)

.. doxygenfunction:: specfem::element::to_string(const property_tag &property)

.. doxygenfunction:: specfem::element::to_string(const boundary_tag &boundary)

.. doxygenfunction:: specfem::element::to_string(const medium_tag &medium, const property_tag &property_tag)

.. doxygenfunction:: specfem::element::to_string(const medium_tag &medium, const property_tag &property_tag, const boundary_tag &boundary_tag)


``specfem::element::attributes``
================================

.. doxygenclass:: specfem::element::attributes
   :members:

Attribute Specializations
--------------------------

.. doxygenclass:: specfem::element::attributes< specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv >
    :members:

.. doxygenclass:: specfem::element::attributes< specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sh >
    :members:

.. doxygenclass:: specfem::element::attributes< specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic >
    :members:

.. doxygenclass:: specfem::element::attributes< specfem::dimension::type::dim2, specfem::element::medium_tag::poroelastic >
    :members:

.. doxygenclass:: specfem::element::attributes< specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv_t >
    :members:

.. doxygenclass:: specfem::element::attributes< specfem::dimension::type::dim2, specfem::element::medium_tag::electromagnetic_te >
    :members:

.. doxygenclass:: specfem::element::attributes< specfem::dimension::type::dim3, specfem::element::medium_tag::elastic >
    :members:

.. doxygenclass:: specfem::element::attributes< specfem::dimension::type::dim3, specfem::element::medium_tag::acoustic >
    :members:

``specfem::element::from_string``
=================================

.. doxygenfunction:: specfem::element::from_string

``specfem::element::is_elastic``
================================

.. doxygentypedef:: specfem::element::is_elastic

``specfem::element::is_electromagnetic``
========================================

.. doxygentypedef:: specfem::element::is_electromagnetic
