#pragma once

namespace specfem {
namespace injection {

/// @brief Named component indices into the @c InjectionFrameBuffer (and the
/// @c out view a provider's sampler writes). This layout is the contract shared
/// by every injection provider (producer), the frame-buffer allocation, and the
/// stiffness-kernel consumer, so it lives in its own header rather than in any
/// one of them.
namespace frame_component {
inline constexpr int displacement_x = 0; ///< u_x
inline constexpr int displacement_y = 1; ///< u_y
inline constexpr int displacement_z = 2; ///< u_z
inline constexpr int traction_x = 3;     ///< t_x (elastic faces)
inline constexpr int traction_y = 4;     ///< t_y (elastic faces)
inline constexpr int traction_z = 5;     ///< t_z (elastic faces)
inline constexpr int potential_rate = 6; ///< χ̇ potential-rate (acoustic faces)
} // namespace frame_component

/// Component count for a displacement-only buffer.
inline constexpr int displacement_component_count = 3;
/// Component count for a displacement + traction buffer (elastic injection).
inline constexpr int displacement_traction_component_count = 6;
/// Component count for a displacement + traction + potential-rate buffer
/// (fluid-solid injection).
inline constexpr int displacement_traction_potential_component_count = 7;

/**
 * @brief Number of frame-buffer components for the requested field options.
 *
 * @param compute_traction Whether traction components (3-5) are stored.
 * @param has_fluid_layer Whether acoustic points — and thus the potential-rate
 *        component (6) — are possible.
 * @return 3 (displacement), 6 (+ traction), or 7 (+ potential-rate).
 */
inline constexpr int injection_component_count(bool compute_traction,
                                               bool has_fluid_layer) {
  if (!compute_traction) {
    return displacement_component_count;
  }
  return has_fluid_layer ? displacement_traction_potential_component_count
                         : displacement_traction_component_count;
}

} // namespace injection
} // namespace specfem
