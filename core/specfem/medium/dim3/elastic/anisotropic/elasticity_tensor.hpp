#pragma once

#include "specfem/coordinate_systems/basis_rotation.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium_physics {

/**
 * @brief Upper triangle of a symmetric elasticity tensor in Voigt order.
 *
 * Components are ordered as c11, c12, c13, c14, c15, c16, c22, c23, c24,
 * c25, c26, c33, c34, c35, c36, c44, c45, c46, c55, c56, c66.
 *
 * @tparam ValueType Scalar value type.
 */
template <typename ValueType>
using elasticity_tensor = Kokkos::Array<ValueType, 21>;

namespace elasticity_tensor_impl {

KOKKOS_INLINE_FUNCTION
constexpr int component_index(int row, int column) {
  if (row > column) {
    const int temporary = row;
    row = column;
    column = temporary;
  }
  return row * 6 - row * (row - 1) / 2 + column - row;
}

} // namespace elasticity_tensor_impl

/**
 * @brief Rotate an elasticity tensor between orthonormal bases.
 *
 * Applies \f$C'_{ijkl} = R_{ia}R_{jb}R_{kc}R_{ld}C_{abcd}\f$ using
 * the Bond matrix for Voigt order (xx, yy, zz, yz, xz, xy). The 21-component
 * upper-triangular storage and engineering-shear convention are preserved.
 *
 * @tparam ValueType Scalar value type.
 * @param input Elasticity tensor in the source basis.
 * @param rotation Orthonormal matrix with target axes as rows and source axes
 * as columns, so vector components transform as \f$v' = Rv\f$.
 * @return Elasticity tensor in the target basis.
 */
template <typename ValueType>
KOKKOS_INLINE_FUNCTION elasticity_tensor<ValueType> rotate_elasticity(
    const elasticity_tensor<ValueType> &input,
    const specfem::coordinate_systems::rotation_matrix<ValueType> &rotation) {
  ValueType stiffness[6][6];
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      stiffness[row][column] =
          input[elasticity_tensor_impl::component_index(row, column)];
    }
  }

  ValueType bond[6][6];
  constexpr int shear_pairs[3][2] = { { 1, 2 }, { 2, 0 }, { 0, 1 } };
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < 3; ++row) {
      bond[row][column] = rotation[row][column] * rotation[row][column];
    }
    for (int row = 0; row < 3; ++row) {
      const int first = shear_pairs[row][0];
      const int second = shear_pairs[row][1];
      bond[row + 3][column] =
          rotation[first][column] * rotation[second][column];
    }
  }
  for (int column = 0; column < 3; ++column) {
    const int first = shear_pairs[column][0];
    const int second = shear_pairs[column][1];
    for (int row = 0; row < 3; ++row) {
      bond[row][column + 3] = static_cast<ValueType>(2) * rotation[row][first] *
                              rotation[row][second];
    }
    for (int row = 0; row < 3; ++row) {
      const int row_first = shear_pairs[row][0];
      const int row_second = shear_pairs[row][1];
      bond[row + 3][column + 3] =
          rotation[row_first][first] * rotation[row_second][second] +
          rotation[row_first][second] * rotation[row_second][first];
    }
  }

  ValueType temporary[6][6] = {};
  for (int column = 0; column < 6; ++column) {
    for (int inner = 0; inner < 6; ++inner) {
      for (int row = 0; row < 6; ++row) {
        temporary[row][column] += stiffness[row][inner] * bond[column][inner];
      }
    }
  }

  elasticity_tensor<ValueType> output{};
  for (int column = 0; column < 6; ++column) {
    for (int inner = 0; inner < 6; ++inner) {
      for (int row = 0; row <= column; ++row) {
        output[elasticity_tensor_impl::component_index(row, column)] +=
            bond[row][inner] * temporary[inner][column];
      }
    }
  }
  return output;
}

/**
 * @brief Construct a radial-frame transversely isotropic elasticity tensor.
 *
 * The Love parameters are
 * \f$A = c_{11} = \rho v_{ph}^{2}\f$,
 * \f$C = c_{33} = \rho v_{pv}^{2}\f$,
 * \f$N = c_{66} = \rho v_{sh}^{2}\f$,
 * \f$L = c_{44} = \rho v_{sv}^{2}\f$, and
 * \f$F = c_{13} = \eta(A - 2L)\f$. Radial transverse isotropy also gives
 * \f$c_{22}=A\f$, \f$c_{12}=A-2N\f$, \f$c_{23}=F\f$, and
 * \f$c_{55}=L\f$; all normal-shear and cross-shear couplings vanish.
 *
 * @tparam ValueType Scalar value type.
 * @param rho Density.
 * @param vpv Vertical P-wave velocity.
 * @param vph Horizontal P-wave velocity.
 * @param vsv Vertical S-wave velocity.
 * @param vsh Horizontal S-wave velocity.
 * @param eta Dimensionless Love anisotropy parameter.
 * @return Radial-frame stiffness tensor in upper-triangular Voigt order.
 */
template <typename ValueType>
KOKKOS_INLINE_FUNCTION elasticity_tensor<ValueType>
love_to_radial_elasticity(const ValueType rho, const ValueType vpv,
                          const ValueType vph, const ValueType vsv,
                          const ValueType vsh, const ValueType eta) {
  const ValueType a = rho * vph * vph;
  const ValueType c = rho * vpv * vpv;
  const ValueType n = rho * vsh * vsh;
  const ValueType l = rho * vsv * vsv;
  const ValueType f = eta * (a - static_cast<ValueType>(2) * l);

  return { a,
           a - static_cast<ValueType>(2) * n,
           f,
           static_cast<ValueType>(0),
           static_cast<ValueType>(0),
           static_cast<ValueType>(0),
           a,
           f,
           static_cast<ValueType>(0),
           static_cast<ValueType>(0),
           static_cast<ValueType>(0),
           c,
           static_cast<ValueType>(0),
           static_cast<ValueType>(0),
           static_cast<ValueType>(0),
           l,
           static_cast<ValueType>(0),
           static_cast<ValueType>(0),
           l,
           static_cast<ValueType>(0),
           n };
}

/**
 * @brief Rotate an elasticity tensor from the local radial frame to Cartesian.
 * @param radial_tensor Tensor whose axes are (theta, phi, radial).
 * @param theta Geocentric colatitude in radians.
 * @param phi Geocentric longitude in radians.
 * @return Tensor in the global Cartesian frame.
 *
 * The angles supplied by the spherical-coordinate cache already satisfy
 * \f$\theta\in[0,\pi]\f$ and \f$\phi\in[0,2\pi)\f$. Unlike globe's legacy
 * `reduce`, this routine does not perturb exact zero angles: the Bond rotation
 * contains no division by \f$\sin(\theta)\f$, so the poles are well-defined.
 */
template <typename ValueType>
KOKKOS_INLINE_FUNCTION elasticity_tensor<ValueType>
rotate_elasticity_radial_to_global(
    const elasticity_tensor<ValueType> &radial_tensor, const ValueType theta,
    const ValueType phi) {
  return rotate_elasticity(
      radial_tensor,
      specfem::coordinate_systems::basis_rotation<
          specfem::coordinate_systems::CartesianFrame,
          specfem::coordinate_systems::GeocentricFrame>(theta, phi));
}

} // namespace specfem::medium_physics
