
Coupled Interface API
=====================

Definition
----------

.. doxygenclass:: specfem::coupled_interface::coupled_interface

Interface
~~~~~~~~~

.. code-block::

  template<typename self_domain, typename coupled_domain>
  class coupled_interface

Parameters
~~~~~~~~~~

.. note::

  Coupled interface template parameters are deduced from the constructor arguments.

.. _domain: ../domain/domain.html

.. |domain| replace:: domain()

* ``self_domain``:

  Primary domain of the edge. This is the domain whose wavefield is updated by methods within this class.

  - |domain|_: An instantiation of a domain class

* ``coupled_domain``:

  Secondary domain of the edge. This is the domain whose wavefield is used to compute coupling interaction.

  - |domain|_: An instantiation of a domain class

Class methods
-------------

.. doxygenclass:: specfem::coupled_interface::coupled_interface
    :members:
    :private-members:

Iterators
---------

.. doxygennamespace:: specfem::compute::coupled_interfaces::iterator
    :members:
