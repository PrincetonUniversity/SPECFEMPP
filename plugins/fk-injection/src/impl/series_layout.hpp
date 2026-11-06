#pragma once

namespace fk_injection::impl {

/// @brief Named indices into the depth-only series set stored by @c FkResult.
/// S0/S1 are kinematic; S2..S4 are stress-related (elastic) or the potential
/// rate (acoustic, S2 only). The depth-series synthesis and @c FkResult::sample
/// share this layout.
inline constexpr int series_horizontal = 0; ///< S0: u_r (elastic), u_x (fluid)
inline constexpr int series_vertical = 1;   ///< S1: u_z
inline constexpr int series_sigma_zz = 2;   ///< S2: omega*p*sigma_zz (elastic)
inline constexpr int series_potential_rate = 2; ///< S2: chi_dot (acoustic)
inline constexpr int series_sigma_rz = 3; ///< S3: -i*omega*p*sigma_rz (elastic)
inline constexpr int series_dx = 4;       ///< S4: omega*p*dx (elastic)
inline constexpr int number_of_series = 5; ///< Count of stored series slots.

} // namespace fk_injection::impl
