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

} // namespace enums
} // namespace specfem
