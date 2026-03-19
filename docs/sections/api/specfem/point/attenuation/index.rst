.. _specfem_point_attenuation:

``specfem::point::attenuation``
================================

.. doxygenstruct:: specfem::point::attenuation
   :members:

Dimension-specific Specializations
------------------------------------

2D PSV Elastic Medium with Constant Isotropic Attenuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: specfem::point::attenuation< specfem::element::dimension_tag::dim2, specfem::element::medium_tag::elastic_psv, specfem::element::attenuation_tag::constant_isotropic, UseSIMD >
   :members:
   :private-members:

3D Elastic Medium with Constant Isotropic Attenuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: specfem::point::attenuation< specfem::element::dimension_tag::dim3, specfem::element::medium_tag::elastic, specfem::element::attenuation_tag::constant_isotropic, UseSIMD >
   :members:
   :private-members:
