Elasticity tensor rotations
===========================

Include ``specfem/medium/dim3/elastic/anisotropic/elasticity_tensor.hpp``
for ``ElasticityTensor<Frame, ValueType>`` and device-callable rotations.
Each tensor carries its coordinate frame in its type. Packed construction
copies 21 coefficients in upper-triangular Voigt order; symmetric access
uses ``tensor(i, j)`` with zero-based indices, and ``tensor[k]`` reads a
packed coefficient in the same order as point properties.

Only ``ElasticityTensor<GeocentricFrame, ValueType>`` supports construction
from ``(rho, vpv, vph, vsv, vsh, eta)``. These Love parameters describe
radial transverse isotropy with symmetry axis r and axes ordered
``(theta, phi, r)``. Fully anisotropic oracle coefficients are already
Cartesian and construct ``ElasticityTensor<CartesianFrame, double>`` directly.

``rotate(tensor, rotation)`` accepts an orthonormal
``specfem::coordinate_systems::BasisMatrix<TargetFrame, SourceFrame, ValueType>``.
Both frames are deduced: the input must have the matrix's source frame,
and the result has its target frame. Equal source and target frames are
rejected. The matrix may describe any orthonormal orientation between
distinct frames. The Bond transformation preserves the upper-triangular
Voigt order ``(xx, yy, zz, yz, xz, xy)`` and its shear factors.

.. code-block:: cpp

    using Cartesian = specfem::coordinate_systems::CartesianFrame;
    using Radial = specfem::coordinate_systems::GeocentricFrame;
    const specfem::medium_physics::ElasticityTensor<Radial, double> radial(
        rho, vpv, vph, vsv, vsh, eta);
    const auto global = specfem::medium_physics::rotate(
        radial, specfem::coordinate_systems::rotation_matrix<Cartesian, Radial>(
                    theta, phi));

For a radial attenuation shift, rotate a global tensor with
``rotation_matrix<GeocentricFrame, CartesianFrame>(theta, phi)``, modify the
radial coefficients, then rotate back with the opposite frame arguments.
See :doc:`../coordinate_systems/basis_rotation` for the basis convention
and behavior at the poles.

.. doxygenclass:: specfem::medium_physics::ElasticityTensor
   :members:

.. doxygenfunction:: specfem::medium_physics::rotate
