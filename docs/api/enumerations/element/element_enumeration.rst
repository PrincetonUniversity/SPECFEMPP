
Element
~~~~~~~

Enumerations that define a type of element.

.. _dim2: dim2.html

.. |dim2| replace:: dim2()

.. _dim3: dim3.html

.. |dim3| replace:: dim3()

.. _elastic: elastic.html

.. |elastic| replace:: elastic()

.. _acoustic: acoustic.html

.. |acoustic| replace:: acoustic()

.. _static_quadrature_points: static_quadrature_points.html

.. |static_quadrature_points| replace:: static_quadrature_points< NGLL >()

.. _isotropic: isotropic.html

.. |isotropic| replace:: isotropic()

.. _boundary_conditions: ../../boundary_conditions/index.html

.. |boundary_conditions| replace:: boundary_conditions

Dimension
_________

The dimension of the element.

- |dim2|_: A two dimensional element.
- |dim3|_: A three dimensional element.

Medium
______

The medium of the element.

- |elastic|_: An elastic element.
- |acoustic|_: An acoustic element.

Medium enumerations to tag the element with a medium.

.. doxygenenum:: specfem::enums::element::type

Quadrature Points
_________________

The quadrature points of the element.

- |static_quadrature_points|_: A static quadrature point set.

Elemental Properties
____________________

The properties of the element. The properties describe any specializations made the implementation.

- Type of element:

  - |isotropic|_: An isotropic element.

Property enumerations to tag the element with a property.

.. doxygenenum:: specfem::enums::element::property_tag

Boundary Conditions
___________________

Defines the boundary conditions on the element. The boundary conditions are used to modify the element's force contribution to the global force vector.

- |boundary_conditions|_: The boundary conditions of the element.

Boundary condition enumerations to tag the element with a boundary condition.

.. doxygenenum:: specfem::enums::element::boundary_tag
