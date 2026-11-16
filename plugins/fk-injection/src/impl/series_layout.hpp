#pragma once

#include <cstddef>

namespace fk_injection::impl {

/// @brief Named indices into the depth-only series set stored by @c
/// SeriesTable. S0/S1 are kinematic; S2..S4 are stress-related (elastic) or the
/// potential rate (acoustic, S2 only). The depth-series synthesis and @c
/// FieldSampler::sample share this layout.
inline constexpr int series_horizontal = 0; ///< S0: u_r (elastic), u_x (fluid)
inline constexpr int series_vertical = 1;   ///< S1: u_z
inline constexpr int series_sigma_zz = 2;   ///< S2: omega*p*sigma_zz (elastic)
inline constexpr int series_potential_rate = 2; ///< S2: chi_dot (acoustic)
inline constexpr int series_sigma_rz = 3; ///< S3: -i*omega*p*sigma_rz (elastic)
inline constexpr int series_dx = 4;       ///< S4: omega*p*dx (elastic)
inline constexpr int number_of_series = 5; ///< Count of stored series slots.

/**
 * @brief Flat index of one spline coefficient in a series array laid out
 *        row-major (row, series, coefficient).
 * @param row               Series-table row
 * @param series            Series slot
 * @param coefficient       Coefficient within the series
 * @param coefficient_count Coefficients per series
 * @return Index into the array
 */
inline std::size_t series_coefficient_index(int row, int series,
                                            int coefficient,
                                            int coefficient_count) {
  return (static_cast<std::size_t>(row) * number_of_series + series) *
             static_cast<std::size_t>(coefficient_count) +
         static_cast<std::size_t>(coefficient);
}

} // namespace fk_injection::impl
