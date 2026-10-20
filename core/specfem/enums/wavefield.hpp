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
 * occupies component 0 only and is written as-is. Scalars are: pressure in any
 * medium, every type in a 2-D SH medium (the field is out-of-plane), and 2-D
 * rotation / intrinsic_rotation / curl (out-of-plane scalars). In 3-D the
 * rotation, intrinsic_rotation and curl fields are axial vectors; under a
 * proper rotation (determinant +1) they transform like polar vectors, so they
 * count as vectors here. This must agree with the per-medium compute_wavefield
 * fills, which leave the unused components zero.
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
  if (dimension == specfem::element::dimension_tag::dim2) {
    if (medium == specfem::element::medium_tag::elastic_sh)
      return false;
    if (type == wavefield::rotation || type == wavefield::intrinsic_rotation ||
        type == wavefield::curl)
      return false;
  }
  return true;
}

} // namespace enums
} // namespace specfem
