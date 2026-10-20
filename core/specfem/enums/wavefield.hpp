#pragma once

#include "specfem/element.hpp"
#include <string>

namespace specfem {
namespace enums {

/**
 * @brief Wavefield component types for different physical quantities.
 *
 * Supports elastic (displacement/velocity/acceleration) and acoustic (pressure)
 * fields.
 */
enum class wavefield {
  displacement,       ///< Displacement field (elastic media)
  velocity,           ///< Velocity field (time derivative of displacement)
  acceleration,       ///< Acceleration field (second time derivative)
  pressure,           ///< Pressure field (acoustic media)
  rotation,           ///< Rotation field (Cosserat media)
  intrinsic_rotation, ///< Intrinsic rotation (micropolar)
  curl                ///< Curl of displacement field
};

/**
 * @brief Convert wavefield component to string.
 * @param wavefield_component Wavefield component type
 * @return String representation ("displacement", "velocity", etc.)
 */
const std::string
to_string(const specfem::enums::wavefield &wavefield_component);

/**
 * @brief Whether a seismogram of this type carries all spatial components.
 *
 * A vector seismogram is rotated into the receiver's local frame; a scalar one
 * occupies component 0 only and is written as-is. Pressure is a scalar in any
 * medium. In 3-D every other type is a vector: rotation, intrinsic_rotation
 * and curl are axial vectors, and under a proper rotation (determinant +1)
 * they transform like polar vectors.
 *
 * In 2-D the kinematic types (displacement, velocity, acceleration) and the
 * axial types (rotation, intrinsic_rotation, curl) are dual. P-SV media carry
 * an in-plane displacement, so its curl and the Cosserat rotation point out of
 * plane and are scalars. SH media carry an out-of-plane displacement, which is
 * the scalar, while its curl and any microrotation lie in plane and are
 * vectors. This must agree with the per-medium compute_wavefield fills, which
 * leave the unused components zero.
 *
 * @param type Seismogram type
 * @param dimension Spatial dimension of the simulation
 * @param medium Medium the receiver element belongs to
 * @return true when the seismogram has `dimension` components
 */
constexpr bool
is_vector_seismogram(const wavefield type,
                     const specfem::element::dimension_tag dimension,
                     const specfem::element::medium_tag medium) {
  if (type == wavefield::pressure)
    return false;
  if (dimension == specfem::element::dimension_tag::dim3)
    return true;
  const bool axial = type == wavefield::rotation ||
                     type == wavefield::intrinsic_rotation ||
                     type == wavefield::curl;
  const bool out_of_plane_displacement =
      medium == specfem::element::medium_tag::elastic_sh;
  return axial == out_of_plane_displacement;
}

} // namespace enums
} // namespace specfem
