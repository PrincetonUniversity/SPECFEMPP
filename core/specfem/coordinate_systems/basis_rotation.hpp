#pragma once

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

/** @brief Rotation matrix with target axes as rows and source axes as columns.
 * @tparam ValueType Scalar value type.
 */
template <typename ValueType>
using rotation_matrix = Kokkos::Array<Kokkos::Array<ValueType, 3>, 3>;

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
KOKKOS_INLINE_FUNCTION rotation_matrix<ValueType>
basis_rotation(const ValueType theta, const ValueType phi) {
  static_assert((std::is_same_v<TargetFrame, CartesianFrame> &&
                 std::is_same_v<SourceFrame, GeocentricFrame>) ||
                    (std::is_same_v<TargetFrame, GeocentricFrame> &&
                     std::is_same_v<SourceFrame, CartesianFrame>),
                "Unsupported source/target frame combination");

  const ValueType cos_theta = Kokkos::cos(theta);
  const ValueType sin_theta = Kokkos::sin(theta);
  const ValueType cos_phi = Kokkos::cos(phi);
  const ValueType sin_phi = Kokkos::sin(phi);
  const ValueType zero = static_cast<ValueType>(0);
  if constexpr (std::is_same_v<TargetFrame, CartesianFrame>) {
    // Columns are the (theta, phi, r) unit vectors in Cartesian components.
    return { { { cos_phi * cos_theta, -sin_phi, cos_phi * sin_theta },
               { sin_phi * cos_theta, cos_phi, sin_phi * sin_theta },
               { -sin_theta, zero, cos_theta } } };
  } else {
    // Rows are the (theta, phi, r) unit vectors in Cartesian components.
    return { { { cos_phi * cos_theta, sin_phi * cos_theta, -sin_theta },
               { -sin_phi, cos_phi, zero },
               { cos_phi * sin_theta, sin_phi * sin_theta, cos_theta } } };
  }
}

} // namespace coordinate_systems
} // namespace specfem
