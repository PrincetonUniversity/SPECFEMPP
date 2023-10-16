.. elemental_receivers_api_doc::

Elemental Receivers API
=======================

Elemental receivers class defines the methods required to compute siesmograms at the receivers for a given siesmogram step. The methods in this class only compute contribution to the seismogram for a single receiver at a single GLL point inside the element where the receiver is located.

Definition
----------

.. doxygenclass:: specfem::domain::impl::receivers::receiver

Interface
~~~~~~~~~

.. code-block::

    template<class dimension, class medium, class quadrature_points_type [, class... Properties]>
    class specfem::domain::impl::receivers::receiver

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

.. _dim2_elastic_static_quadrature_points_isotropic: receivers_dim2_elastic_static_quadrature_points_isotropic.html

.. |dim2_elastic_static_quadrature_points_isotropic| replace:: receiver< |dim2|_, |elastic|_, |static_quadrature_points|_, |isotropic|_ >()

.. _dim2_acoustic_static_quadrature_points_isotropic: receivers_dim2_acoustic_static_quadrature_points_isotropic.html

.. |dim2_acoustic_static_quadrature_points_isotropic| replace:: receiver< |dim2|_, |acoustic|_, |static_quadrature_points|_, |isotropic|_ >()

* 2D elastic isotropic elements:

  - |dim2_elastic_static_quadrature_points_isotropic|_: Receiver located in a two dimensional elastic element with static quadrature points and isotropic material properties.

* 2D acoustic isotropic elements:

  - |dim2_acoustic_static_quadrature_points_isotropic|_: Receiver located in a two dimensional acoustic element with static quadrature points and isotropic material properties.
