#pragma once

#include "specfem/coordinate_systems/basis_rotation.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium_physics_impl {

/**
 * @brief Packed storage shared by elasticity tensors in all coordinate frames.
 * @tparam ValueType Scalar value type.
 */
template <typename ValueType> class ElasticityTensorBase {
public:
  KOKKOS_INLINE_FUNCTION ElasticityTensorBase() = default;

  /**
   * @brief Copy 21 coefficients in upper-triangular Voigt order.
   * @param coefficients Packed c11, c12, ..., c56, c66 coefficients.
   */
  KOKKOS_INLINE_FUNCTION explicit ElasticityTensorBase(
      const ValueType *coefficients) {
    for (int k = 0; k < 21; ++k) {
      coefficients_[k] = coefficients[k];
    }
  }

  /** @brief Symmetric Voigt access with zero-based indices. */
  KOKKOS_INLINE_FUNCTION ValueType &operator()(int row, int column) {
    return coefficients_[component_index(row, column)];
  }
  /** @brief Symmetric Voigt access with zero-based indices. */
  KOKKOS_INLINE_FUNCTION ValueType operator()(int row, int column) const {
    return coefficients_[component_index(row, column)];
  }
  /** @brief Packed access matching point properties' c11..c66 order. */
  KOKKOS_INLINE_FUNCTION ValueType operator[](int k) const {
    return coefficients_[k];
  }

private:
  KOKKOS_INLINE_FUNCTION static constexpr int component_index(int row,
                                                              int column) {
    if (row > column) {
      const int temporary = row;
      row = column;
      column = temporary;
    }
    return row * 6 - row * (row - 1) / 2 + column - row;
  }

  ValueType coefficients_[21] = {};
};

} // namespace specfem::medium_physics_impl

namespace specfem::medium_physics {

/**
 * @brief Symmetric elasticity tensor expressed in the axes of Frame.
 *
 * Components are ordered as c11, c12, c13, c14, c15, c16, c22, c23, c24,
 * c25, c26, c33, c34, c35, c36, c44, c45, c46, c55, c56, c66.
 *
 * @tparam Frame Coordinate frame of the tensor components.
 * @tparam ValueType Scalar value type.
 */
template <typename Frame, typename ValueType>
class ElasticityTensor
    : public specfem::medium_physics_impl::ElasticityTensorBase<ValueType> {
public:
  using specfem::medium_physics_impl::ElasticityTensorBase<
      ValueType>::ElasticityTensorBase;
  KOKKOS_INLINE_FUNCTION ElasticityTensor() = default;
};

/**
 * @brief Radial elasticity tensor with axes ordered (theta, phi, r).
 * @tparam ValueType Scalar value type.
 */
template <typename ValueType>
class ElasticityTensor<specfem::coordinate_systems::GeocentricFrame, ValueType>
    : public specfem::medium_physics_impl::ElasticityTensorBase<ValueType> {
public:
  using specfem::medium_physics_impl::ElasticityTensorBase<
      ValueType>::ElasticityTensorBase;
  KOKKOS_INLINE_FUNCTION ElasticityTensor() = default;

  /**
   * @brief Construct radial transverse isotropy from Love parameters.
   *
   * \f$A = \rho v_{ph}^2,\ C = \rho v_{pv}^2,\ N = \rho v_{sh}^2,
   * L = \rho v_{sv}^2,\ F = \eta(A - 2L)\f$.
   * The symmetry axis is r; all normal-shear and cross-shear couplings vanish.
   *
   * @param rho Density.
   * @param vpv Vertical P-wave velocity.
   * @param vph Horizontal P-wave velocity.
   * @param vsv Vertical S-wave velocity.
   * @param vsh Horizontal S-wave velocity.
   * @param eta Dimensionless Love anisotropy parameter.
   */
  KOKKOS_INLINE_FUNCTION
  ElasticityTensor(const ValueType rho, const ValueType vpv,
                   const ValueType vph, const ValueType vsv,
                   const ValueType vsh, const ValueType eta) {
    const ValueType a = rho * vph * vph;
    const ValueType c = rho * vpv * vpv;
    const ValueType n = rho * vsh * vsh;
    const ValueType l = rho * vsv * vsv;
    const ValueType f = eta * (a - static_cast<ValueType>(2) * l);
    auto &tensor = *this;
    tensor(0, 0) = tensor(1, 1) = a;
    tensor(0, 1) = a - static_cast<ValueType>(2) * n;
    tensor(0, 2) = tensor(1, 2) = f;
    tensor(2, 2) = c;
    tensor(3, 3) = tensor(4, 4) = l;
    tensor(5, 5) = n;
  }
};

/**
 * @brief Rotate an elasticity tensor between distinct orthonormal bases.
 *
 * Applies \f$C'_{ijkl} = R_{ia}R_{jb}R_{kc}R_{ld}C_{abcd}\f$ using
 * the Bond matrix for Voigt order (xx, yy, zz, yz, xz, xy). The 21-component
 * upper-triangular storage and engineering-shear convention are preserved.
 * Both frames are deduced; the tensor must match the matrix's source frame.
 *
 * @tparam TargetFrame Coordinate frame of the result.
 * @tparam SourceFrame Coordinate frame of the input.
 * @tparam ValueType Scalar value type.
 * @param input Elasticity tensor in the source basis.
 * @param rotation Orthonormal matrix with target axes as rows and source axes
 * as columns, so vector components transform as \f$v' = Rv\f$.
 * @return Elasticity tensor in the target basis.
 */
template <typename TargetFrame, typename SourceFrame, typename ValueType>
  requires(!std::is_same_v<TargetFrame, SourceFrame>)
KOKKOS_INLINE_FUNCTION ElasticityTensor<TargetFrame, ValueType>
rotate(const ElasticityTensor<SourceFrame, ValueType> &input,
       const specfem::coordinate_systems::BasisMatrix<TargetFrame, SourceFrame,
                                                      ValueType> &rotation) {
  ValueType stiffness[6][6];
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      stiffness[row][column] = input(row, column);
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

  ElasticityTensor<TargetFrame, ValueType> output{};
  for (int column = 0; column < 6; ++column) {
    for (int inner = 0; inner < 6; ++inner) {
      for (int row = 0; row <= column; ++row) {
        output(row, column) += bond[row][inner] * temporary[inner][column];
      }
    }
  }
  return output;
}

} // namespace specfem::medium_physics
