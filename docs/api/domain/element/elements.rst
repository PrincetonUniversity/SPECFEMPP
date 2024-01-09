.. _element_api_documentation::

Element API
===========

Elements are the building blocks of the compute infratructure within a domain. Each instance of an element class respresents a single spectral element within the domain. The methods within this class compute elemental contributions at a single quadrature within the element.

Definition
----------

.. doxygenclass:: specfem::domain::impl::elements::element

Interface
~~~~~~~~~

.. code-block::

    template<class dimension, class medium, class quadrature_points_type , class Properties, class boundary_conditions>
    class specfem::domain::impl::elements::element

Parameters
~~~~~~~~~~

.. _dim2: ../../enumerations/element/dim2.html

.. |dim2| replace:: dim2()

.. _dim3: ../../enumerations/element/dim3.html

.. |dim3| replace:: dim3()

.. _elastic: ../../enumerations/element/elastic.html

.. |elastic| replace:: elastic()

.. _acoustic: ../../enumerations/element/acoustic.html

.. |acoustic| replace:: acoustic()

.. _static_quadrature_points: ../../enumerations/element/static_quadrature_points.html

.. |static_quadrature_points| replace:: static_quadrature_points< NGLL >()

.. _isotropic: ../../enumerations/element/isotropic.html

.. |isotropic| replace:: isotropic()

.. _boundary_conditions: ../boundary_conditions/boundary_conditions.html

.. _dirichlet: ../boundary_conditions/dirichlet.html

.. |dirichlet| replace:: dirichlet()

.. _stacey: ../boundary_conditions/stacey.html

.. |stacey| replace:: stacey()

.. _none: ../boundary_conditions/none.html

.. |none| replace:: none()

.. _composite: ../boundary_conditions/composite.html

.. |composite| replace:: composite()

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

* ``boundary_conditions``:

  The boundary conditions to be applied to the element. The boundary conditions modify the contribution of element's force vector to the global force vector.

.. note::
    An element on the boundary is not Boundary conditions are not specified, the element will be assumed to have neumann boundary condition.

  - |dirichlet|_: A Dirichlet boundary condition.
  - |stacey|_: A Stacey boundary condition.
  - |none|_ (default): No boundary condition.
  - |composite|_: A composite boundary condition.


.. warning::

  Implemetations exists for only for the specializations defined below.

Template specializations
-------------------------

.. _dim2_elastic_static_quadrature_points_isotropic: elements_dim2_elastic_static_quadrature_points_isotropic.html

.. |dim2_elastic_static_quadrature_points_isotropic| replace:: element< |dim2|_, |elastic|_, |static_quadrature_points|_, |isotropic|_, typename |boundary_conditions|_ >()

.. _dim2_acoustic_static_quadrature_points_isotropic: elements_dim2_acoustic_static_quadrature_points_isotropic.html

.. |dim2_acoustic_static_quadrature_points_isotropic| replace:: element< |dim2|_, |acoustic|_, |static_quadrature_points|_, |isotropic|_, typename |boundary_conditions|_ >()

* 2D elastic isotropic elements:

  - |dim2_elastic_static_quadrature_points_isotropic|_: A two dimensional elastic element with static quadrature points and isotropic material properties.

* 2D acoustic isotropic elements:

  - |dim2_acoustic_static_quadrature_points_isotropic|_: A two dimensional acoustic element with static quadrature points and isotropic material properties.
