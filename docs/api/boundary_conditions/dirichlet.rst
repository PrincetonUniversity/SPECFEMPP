.. _dirichlet_bc:

Dirichlet boundary conditions
==============================

Definition
----------

.. doxygenclass:: specfem::enums::boundary_conditions::dirichlet

Interface
---------

.. codeblock::

  template <class dimension, class medium, class property, class quadrature_points_type>
  class dirichlet

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

Template Implementation
-----------------------

.. _dirichlet_implementation: dirichlet_implementation.html

.. |dirichlet_implementation| replace:: dirichlet< typename dimension , typename medium , typename property , typename quadrature_points >()

* Dirichlet implementation of various elements

    - |dirichlet_implementation|_
