.. _domain::

Domain API
==========

Definition
----------

.. doxygenclass:: specfem::domain::domain

Interface
.........

.. code-block::

    template <class medium, class quadrature_points_type>
    class specfem::domain::domain

Parameters
..........

.. _elastic:: elastic.html

.. |elastic| replace:: "elastic()"

.. _acoustic:: acoustic.html

.. |acoustic| replace:: "acoustic()"

.. _static_quadrature_points:: static_quadrature_points.html

.. |static_quadrature_points| replace:: "static_quadrature_points()"

* ``medium``:

  The medium of the element.

  - |elastic|: An elastic element.
  - |acoustic|: An acoustic element.

* ``quadrature_points_type``:

  The quadrature points of the element.

  - |static_quadrature_points|: A static quadrature point set.

Class members
-------------

.. doxygenclass:: specfem::domain::domain
    :members:
    :private-members:
