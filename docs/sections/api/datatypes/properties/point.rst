
.. _datatype_point_material_properties:

Point Properties
================

Datatype used to store properties at quadrature point.

.. doxygenstruct:: specfem::point::properties
    :members:

Implementation Details
----------------------

.. doxygenstruct:: specfem::point::properties< specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic, specfem::element::property_tag::isotropic, UseSIMD >
        :members:

.. doxygenstruct:: specfem::point::properties< specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv, specfem::element::property_tag::isotropic, UseSIMD >
    :members:

.. doxygenstruct:: specfem::point::properties< specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sh, specfem::element::property_tag::isotropic, UseSIMD >
    :members:

.. doxygenstruct:: specfem::point::properties< specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv, specfem::element::property_tag::anisotropic, UseSIMD >
    :members:

.. doxygenstruct:: specfem::point::properties< specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sh, specfem::element::property_tag::anisotropic, UseSIMD >
    :members:

.. doxygenstruct:: specfem::point::properties< specfem::dimension::type::dim2, specfem::element::medium_tag::poroelastic, specfem::element::property_tag::isotropic, UseSIMD >
        :members:
