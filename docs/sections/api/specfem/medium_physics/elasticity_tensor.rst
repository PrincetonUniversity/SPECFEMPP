Elasticity tensor rotations
===========================

Include ``specfem/medium/dim3/elastic/anisotropic/elasticity_tensor.hpp``
for 21-component elasticity tensors and device-callable rotations.

``rotate_elasticity(tensor, rotation)`` accepts an orthonormal 3-by-3
``specfem::coordinate_systems::rotation_matrix<ValueType>``. It applies the
fourth-order tensor transformation using a Bond matrix, preserving the
upper-triangular Voigt order ``(xx, yy, zz, yz, xz, xy)`` and its shear factors.
Rows of the rotation matrix identify target axes; columns identify source
axes. The matrix may describe any orthonormal orientation.

``rotate_elasticity_radial_to_global`` is a convenience wrapper that uses
:doc:`../coordinate_systems/basis_rotation` with local axes ordered
``(theta, phi, r)``. Love-parameter construction remains in the radial frame.
For the inverse, pass
``basis_rotation<GeocentricFrame, CartesianFrame>(theta, phi)`` to
``rotate_elasticity``.

.. doxygenfunction:: specfem::medium_physics::rotate_elasticity

.. doxygenfunction:: specfem::medium_physics::rotate_elasticity_radial_to_global

.. doxygenfunction:: specfem::medium_physics::love_to_radial_elasticity
