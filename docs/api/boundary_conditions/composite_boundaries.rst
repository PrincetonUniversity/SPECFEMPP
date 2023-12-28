.. _composite_boundaries:

Composite Boundaries
====================

Composite boundaries are a special type of boundary that is used to enforce a combination of multiple boundary conditions. For example, a composite boundary can be used to enforce a Dirichlet boundary condition on the top boundary of an element and a stacey ABC on the right/left boundaries of the element.

Definition
----------

.. doxygenclass:: specfem::enums::boundary_conditions::composite_boundary

Interface
---------

.. codeblock::

    template <typename... BC>
    class composite_boundary;

Parameters
----------

.. _stacey: stacey.html

.. |stacey| replace:: stacey()

.. _dirichlet: dirichlet.html

.. |dirichlet| replace:: dirichlet()

.. _none: none.html

.. |none| replace:: none()

* ``BC...``: A variadic list of 2 or more boundary conditions. The boundary conditions must be one of the following:

    - |stacey|_ : Stacey absorbing boundary condition
    - |dirichlet|_ : Dirichlet boundary condition
    - |none|_ : No boundary condition

.. note::

    Template specializations are provided for the combination of boundary conditions listed below.

Template Specializations
------------------------

.. _stacey_dirichlet: stacey_dirichlet_implementation.html

.. |stacey_dirichlet| replace:: composite_boundary< |stacey|_ , |dirichlet|_ >

* Composite boundary condition enforcing a stacey ABC on one edge and a Dirichlet boundary condition on another edge.

    - |stacey_dirichlet|_
