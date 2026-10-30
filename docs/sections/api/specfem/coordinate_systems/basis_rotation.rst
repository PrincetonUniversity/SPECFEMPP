Basis rotations
===============

Include ``specfem/coordinate_systems/basis_rotation.hpp`` to construct a
device-callable change of basis. The target frame is the first template
argument; the source frame is the second. The returned
``BasisMatrix<TargetFrame, SourceFrame, ValueType>`` retains both frames,
so tensor rotations can check that the source frame matches. Equal source
and target frames are rejected. Matrix storage uses a rank-2
``RegisterArray`` with ``Kokkos::layout_right``: literal coefficients are
listed row by row, and components are accessed as ``rotation(i, j)`` with
debug bounds checks. Matrices support zero-initialized default construction
and explicit construction from nine coefficients of exactly ``ValueType``.
Scalar-fill and pointer conversions are not exposed, and coefficient
construction rejects mixed scalar types, including double-to-float conversion.

.. code-block:: cpp

    const auto rotation = specfem::coordinate_systems::rotation_matrix<
        specfem::coordinate_systems::CartesianFrame,
        specfem::coordinate_systems::GeocentricFrame>(theta, phi);
    const auto global =
        specfem::medium_physics::rotate(radial, rotation);

The matrix maps source vector components to target components as
:math:`v'_i = R_{ij}v_j`. Cartesian axes are ordered :math:`(x,y,z)`;
the local orthonormal axes are ordered :math:`(\theta,\phi,r)`.
Note that this differs from the field order :math:`(r,\theta,\phi)` of
``geocentric_coordinates``.
Angles are geocentric colatitude and longitude in radians. Reversing the
frame arguments returns the transpose. The inverse specialization calls
``transpose`` on the forward rotation, so both directions share one formula.
Transposing a ``BasisMatrix`` also swaps its frame tags. At the poles, longitude selects
the tangential axes; the angles are used without perturbation.

This operation changes vector or tensor components at a point. Position
coordinate conversions use :doc:`to`.

.. doxygenstruct:: specfem::coordinate_systems::CartesianFrame

.. doxygenstruct:: specfem::coordinate_systems::GeocentricFrame

.. doxygenstruct:: specfem::coordinate_systems::BasisMatrix
   :members:

The ``RotationMatrix<TargetFrame, SourceFrame>`` callable has specializations
only for the two supported directions. The device-annotated
``rotation_matrix`` function constructs this callable locally; it does not
access a namespace-level callable object from device code. Direct calls such
as ``RotationMatrix<CartesianFrame, GeocentricFrame>{}(theta, phi)`` are
also supported.

.. doxygenstruct:: specfem::coordinate_systems::RotationMatrix

.. doxygenfunction:: specfem::coordinate_systems::transpose

.. doxygenfunction:: specfem::coordinate_systems::rotation_matrix
