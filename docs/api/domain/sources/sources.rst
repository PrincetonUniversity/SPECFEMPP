.. elemental_sources_api_documentation

Elemental Sources API
=====================

Elemental sources class defines the methods required to compute the contribution of a source to the global acceleration at a given timestep. The methods in this class only compute the contributions of a single source for a single GLL point in the element where the source is located.

Definition
----------

.. doxygenclass:: specfem::domain::impl::sources::source

Interface
~~~~~~~~~

.. code-block::

    template<class dimension, class medium, class quadrature_points_type [, class... Properties]>
    class specfem::domain::impl::sources::source

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

* ``Properties``:

  The properties of the element. The properties describe any specializations made the implementation.

  - Type of element:

    - |isotropic|_: An isotropic element.

.. warning::

  Implemetations exists for only for the specializations defined below.

Template specializations
-------------------------

.. _dim2_elastic_static_quadrature_points_isotropic: sources_dim2_elastic_static_quadrature_points_isotropic.html

.. |dim2_elastic_static_quadrature_points_isotropic| replace:: source< |dim2|_, |elastic|_, |static_quadrature_points|_, |isotropic|_ >()

.. _dim2_acoustic_static_quadrature_points_isotropic: sources_dim2_acoustic_static_quadrature_points_isotropic.html

.. |dim2_acoustic_static_quadrature_points_isotropic| replace:: source< |dim2|_, |acoustic|_, |static_quadrature_points|_, |isotropic|_ >()

* 2D elastic isotropic elements:

  - |dim2_elastic_static_quadrature_points_isotropic|_: Source located in a two dimensional elastic element with static quadrature points and isotropic material properties.

* 2D acoustic isotropic elements:

  - |dim2_acoustic_static_quadrature_points_isotropic|_: Source located in a two dimensional acoustic element with static quadrature points and isotropic material properties.
