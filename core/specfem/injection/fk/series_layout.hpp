#pragma once

namespace specfem {
namespace injection {
namespace fk_impl {

/// @brief Named indices into the depth-only series set: the rows of
/// @c FkResult::series_coefficients and of @c SeriesSpectrum::values. S0/S1 are
/// kinematic; S2..S4 are stress-related (elastic) or pressure (acoustic, S2
/// only). This layout is the contract shared by the depth-series synthesis and
/// the device sampler, so it lives in its own header rather than in either.
inline constexpr int series_horizontal = 0; ///< S0: u_r (elastic), u_x (fluid)
inline constexpr int series_vertical = 1;   ///< S1: u_z
inline constexpr int series_sigma_zz = 2;   ///< S2: omega*p*sigma_zz / pressure
inline constexpr int series_sigma_rz = 3; ///< S3: -i*omega*p*sigma_rz (elastic)
inline constexpr int series_dx = 4;       ///< S4: omega*p*dx (elastic)
inline constexpr int number_of_series = 5; ///< Count of stored series slots.

} // namespace fk_impl
} // namespace injection
} // namespace specfem
