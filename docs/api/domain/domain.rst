.. _domain::

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

.. _dim2: :ref:`<specfem_enums_element_dimension_dim2>`

.. |dim2| replace:: dim2()

.. _dim3: :ref:`<specfem_enums_element_dimension_dim3>`

.. |dim3| replace:: dim3()

.. _elastic: :ref:`<specfem_enums_element_medium_elastic>`

.. |elastic| replace:: elastic()

.. _acoustic: :ref:`<specfem_enums_element_medium_acoustic>`

.. |acoustic| replace:: acoustic()

.. _static_quadrature_points: :ref:`<specfem_enums_element_quadrature_static_quadrature_points>`

.. |static_quadrature_points| replace:: static_quadrature_points< NGLL >()

.. _isotropic: :ref:`<specfem_enums_element_properties_isotropic>`

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
