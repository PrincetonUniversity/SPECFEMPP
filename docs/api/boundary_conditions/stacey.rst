.. _stacey_ABCs::

Stacey ABCs
============

Definition
----------

.. doxygenclass:: specfem::enums::boundary_conditions::stacey

Interface
---------

.. codeblock::

    template <class dimension, class medium, class property, class quadrature_points_type>
    class stacey

Parameters
----------

.. _dim2: ../enumerations/element/dim2.html

.. |dim2| replace:: dim2()

.. _dim3: ../enumerations/element/dim3.html

.. |dim3| replace:: dim3()

.. _elastic: ../enumerations/element/elastic.html

.. |elastic| replace:: elastic()

.. _acoustic: ../enumerations/element/acoustic.html

.. |acoustic| replace:: acoustic()

.. _static_quadrature_points: ../enumerations/element/static_quadrature_points.html

.. |static_quadrature_points| replace:: static_quadrature_points< NGLL >()

.. _isotropic: ../enumerations/element/isotropic.html

.. |isotropic| replace:: isotropic()

* ``dimension``:

  The dimension of the element.

  - |dim2|_: A two dimensional element.
  - |dim3|_: A three dimensional element.

* ``medium``:

  The medium of the element.

  - |elastic|_: An elastic element.
  - |acoustic|_: An acoustic element.

* ``quadrature_points_type``:

  The quadrature points of the element.

  - |static_quadrature_points|_: A static quadrature point set.

* ``properties``:

  The properties of the element. The properties describe any specializations made the implementation.

  - Type of element:

    - |isotropic|_: An isotropic element.

.. warning::

    The Stacey ABCs are only implemented for the combination of parameters listed below.

Template Specializations
------------------------

.. _stacey_dim2_elastic: stacey_dim2_elastic_implementation.html

.. |stacey_dim2_elastic| replace:: stacey< |dim2|_, |elastic|_, typename property , typename quadrature_points >()

.. _stacey_dim2_acoustic: stacey_dim2_acoustic_implementation.html

.. |stacey_dim2_acoustic| replace:: stacey< |dim2|_, |acoustic|_, typename property , typename quadrature_points >()

* Stacey ABCs for a two dimensional elements.

    - |stacey_dim2_elastic|_

* Stacey ABCs for a two dimensional isotropic acoustic element with a static quadrature point set.

    - |stacey_dim2_acoustic|_
