
Edge API
========

Edges are used as building blocks for any coupled interface. The methods presented within edge class evaluate the coupling physics at a single GLL point within the edge

Definition
----------

.. doxygenclass:: specfem::coupled_interface::impl::edges::edge

Interface
~~~~~~~~~

.. code-block::

  template<typename self_domain, typename coupled_domain>
  class edge

Parameters
~~~~~~~~~~

.. note::

  template parameters are deduced from the constructor arguments by the compiler

.. _domain: ../../domain/domain.html

.. |domain| replace:: domain()

* ``self_domain``:

  Primary domain of the edge. This is the domain whose wavefield is updated by methods within this class.

  - |domain|_: An instantiation of a domain class

* ``coupled_domain``:

  Secondary domain of the edge. This is the domain whose wavefield is used to compute coupling interaction.

  - |domain|_: An instantiation of a domain class

.. warning::

  Implemetations exists for only for the specializations defined below.

Template Specializations
------------------------

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

.. |static_elastic_domain| replace:: domain< |elastic|_, |static_quadrature_points|_ >()

.. |static_acoustic_domain| replace:: domain< |acoustic|_, |static_quadrature_points|_ >()

.. _elastic_acoustic_edge: elastic_acoustic/elastic_acoustic_edge.html

.. |elastic_acoustic_edge| replace:: edge< |static_elastic_domain|, |static_acoustic_domain| >()

.. _acoustic_elastic_edge: elastic_acoustic/acoustic_elastic_edge.html

.. |acoustic_elastic_edge| replace:: edge< |static_acoustic_domain|, |static_elastic_domain| >()

* Elastic-Acoustic Edges

  - |elastic_acoustic_edge|_: Elastic acoustic edge with elastic domain as primary domain and acoustic domain as secondary domain
  - |acoustic_elastic_edge|_: Acoustic elastic edge with acoustic domain as primary domain and elastic domain as secondary domain
