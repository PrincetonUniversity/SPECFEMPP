
Kernels
=======

The ``kernels`` namespace is used to store Kokkos kernel implementations. The kernel implemetation describes the parallelism required to execute the computations within elements, elemental sources and elemental receivers.

Kernels storage class
---------------------

Interface
~~~~~~~~~

.. code-block::

  template <class medium, class quadrature_points_type>
  class specfem::domain::impl::kernels::kernels

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

Class Description
~~~~~~~~~~~~~~~~~

.. _kernels: kernels.html

.. |kernels| replace:: kernels()

- |kernels|_

Kernel implementations
----------------------

.. _elemental_kernels: elemental_kernels.html

.. |elemental_kernels| replace:: elemental_kernels()

.. _source_kernels: source_kernel.html

.. |source_kernels| replace:: source_kernels()

.. _receiver_kernels: receiver_kernels.html

.. |receiver_kernels| replace:: receiver_kernels()

- |elemental_kernels|_
- |source_kernels|_
- |receiver_kernels|_
