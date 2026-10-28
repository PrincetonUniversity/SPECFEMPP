#pragma once

#include "specfem/datatype/register_array.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace coordinate_systems {

/** @brief Global Cartesian orthonormal basis (x, y, z). */
struct CartesianFrame {};

/**
 * @brief Local geocentric orthonormal basis with axes ordered (theta, phi, r).
 *
 * The axis order differs from the field order (r, theta, phi) of
 * @ref geocentric_coordinates.
 */
struct GeocentricFrame {};

/**
 * @brief Orthonormal matrix mapping source components to target components.
 * @tparam TargetFrame Frame of the rows.
 * @tparam SourceFrame Frame of the columns; must differ from TargetFrame.
 * @tparam ValueType Scalar value type.
 */
template <typename TargetFrame, typename SourceFrame, typename ValueType>
  requires(!std::is_same_v<TargetFrame, SourceFrame>)
struct BasisMatrix
    : public specfem::datatype::RegisterArray<
          ValueType, Kokkos::extents<std::size_t, 3, 3>, Kokkos::layout_right> {
  using specfem::datatype::RegisterArray<ValueType,
                                         Kokkos::extents<std::size_t, 3, 3>,
                                         Kokkos::layout_right>::RegisterArray;
};

/**
 * @brief Invert an orthonormal basis matrix and swap its frame tags.
 * @param rotation Matrix mapping source components to target components.
 * @return Transposed matrix mapping target components to source components.
 */
template <typename TargetFrame, typename SourceFrame, typename ValueType>
KOKKOS_INLINE_FUNCTION BasisMatrix<SourceFrame, TargetFrame, ValueType>
transpose(const BasisMatrix<TargetFrame, SourceFrame, ValueType> &rotation) {
  BasisMatrix<SourceFrame, TargetFrame, ValueType> inverse;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      inverse(row, column) = rotation(column, row);
    }
  }
  return inverse;
}

/**
 * @brief Construct a change of basis for a supported pair of frames.
 *
 * Only the Cartesian/Geocentric specializations are defined.
 * @tparam TargetFrame Frame of the result's rows.
 * @tparam SourceFrame Frame of the result's columns.
 */
template <typename TargetFrame, typename SourceFrame> struct RotationMatrix;

/** @brief Geocentric (theta, phi, r) to Cartesian (x, y, z). */
template <> struct RotationMatrix<CartesianFrame, GeocentricFrame> {
  /** @brief Construct the forward rotation from colatitude and longitude. */
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION BasisMatrix<CartesianFrame, GeocentricFrame, ValueType>
  operator()(const ValueType theta, const ValueType phi) const {
    const ValueType cos_theta = Kokkos::cos(theta);
    const ValueType sin_theta = Kokkos::sin(theta);
    const ValueType cos_phi = Kokkos::cos(phi);
    const ValueType sin_phi = Kokkos::sin(phi);
    // Row-major storage: columns are the (theta, phi, r) Cartesian unit
    // vectors.
    return { cos_phi * cos_theta, -sin_phi,       cos_phi * sin_theta,
             sin_phi * cos_theta, cos_phi,        sin_phi * sin_theta,
             -sin_theta,          ValueType{ 0 }, cos_theta };
  }
};

/** @brief Cartesian to geocentric, derived from the forward rotation. */
template <> struct RotationMatrix<GeocentricFrame, CartesianFrame> {
  /** @brief Construct the inverse rotation from colatitude and longitude. */
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION BasisMatrix<GeocentricFrame, CartesianFrame, ValueType>
  operator()(const ValueType theta, const ValueType phi) const {
    return transpose(
        RotationMatrix<CartesianFrame, GeocentricFrame>{}(theta, phi));
  }
};

/**
 * @brief Construct the change of basis between geocentric and Cartesian frames.
 *
 * Vector components transform as \f$v^{target}_i = R_{ij}v^{source}_j\f$.
 * The local basis is ordered (theta, phi, r), with unit-length axes; this is
 * a basis rotation, not a conversion of position coordinates. The inverse
 * rotation is the transpose. At a pole, phi selects the tangential axes;
 * no angle perturbation is applied.
 *
 * @tparam TargetFrame CartesianFrame or GeocentricFrame.
 * @tparam SourceFrame The opposite frame to TargetFrame.
 * @tparam ValueType Scalar value type (deduced).
 * @param theta Geocentric colatitude in radians.
 * @param phi Geocentric longitude in radians.
 * @return Orthonormal rotation mapping source components to target components.
 */
template <typename TargetFrame, typename SourceFrame, typename ValueType>
  requires requires(const ValueType theta, const ValueType phi) {
    RotationMatrix<TargetFrame, SourceFrame>{}(theta, phi);
  }
KOKKOS_INLINE_FUNCTION BasisMatrix<TargetFrame, SourceFrame, ValueType>
rotation_matrix(const ValueType theta, const ValueType phi) {
  // Construct the stateless callable locally, including in device code.
  return RotationMatrix<TargetFrame, SourceFrame>{}(theta, phi);
}

} // namespace coordinate_systems
} // namespace specfem
