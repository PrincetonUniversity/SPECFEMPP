.. _boundary_conditions:

Boundary conditions
-------------------

Boundary conditions class is used as a template parameter of :doxygenclass:`specfem::domain::impl::elements::element` class. The approach for applying boundary conditions is defined more in detail in the :ref:`boundary_conditions` section.

Interface
~~~~~~~~~

Interface for various types of boundary conditions

.. codeblock::

    template <class dimension, class medium, class property, class quadrature_points_type>
    class (boundary_condition_type)

Types of boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following boundary conditions are implemented:

.. toctree::
    :maxdepth: 1

    stacey
    dirichlet
    none
    composite_boundaries
