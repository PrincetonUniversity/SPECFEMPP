.. _domain:

Domain API
----------

Definition
==========

.. doxygenclass:: specfem::domain::domain

Interface
~~~~~~~~~

.. code-block::

  template <class medium, class quadrature_points_type>
  class specfem::domain::domain

Parameters
~~~~~~~~~~

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

* ``medium``:

  The medium of the element.

  - |elastic|_: An elastic element.
  - |acoustic|_: An acoustic element.

* ``quadrature_points_type``:

  The quadrature points of the element.

  - |static_quadrature_points|_: A static quadrature point set.

Class members
=============

.. doxygenclass:: specfem::domain::domain
    :members:
    :private-members:
