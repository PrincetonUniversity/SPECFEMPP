
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

.. _domain: domain.html

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

.. |static_elastic_domain| replace:: domain< |elastic|_, |static_quadrature_points|_ >()

.. |static_acoustic_domain| replace:: domain< |acoustic|_, |static_quadrature_points|_ >()

.. _elastic_acoustic_edge: :ref:`<specfem_enums_element_edge_elastic_acoustic>`

.. |elastic_acoustic_edge| replace:: edge< |static_elastic_domain|, |static_acoustic_domain|>()

.. _acoustic_elastic_edge: :ref:`<specfem_enums_element_edge_acoustic_elastic>`

.. |acoustic_elastic_edge| replace:: edge< |static_acoustic_domain|, |static_elastic_domain| >()

* Elastic-Acoustic Edges

  - |elastic_acoustic_edge|_: Elastic acoustic edge with elastic domain as primary domain and acoustic domain as secondary domain
  - |acoustic_elastic_edge|_: Acoustic elastic edge with acoustic domain as primary domain and elastic domain as secondary domain
