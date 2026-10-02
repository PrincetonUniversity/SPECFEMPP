#pragma once

namespace specfem::globe {

/** @brief SPECFEM3D_GLOBE radial-zone flags stored as @c idoubling. */
enum class radial_flag : int {
  crust = 1,
  moho_to_80 = 2,
  depth_80_to_220 = 3,
  depth_220_to_670 = 4,
  mantle_normal = 5,
  outer_core_normal = 6,
  inner_core_normal = 7,
  middle_central_cube = 8,
  bottom_central_cube = 9,
  top_central_cube = 10,
  fictitious_cube = 11
};

/**
 * @brief Whether a radial-zone flag identifies a physical central-cube element.
 * @param idoubling Raw SPECFEM3D_GLOBE radial-zone flag.
 * @return True for the middle, bottom, and top central-cube zones.
 */
constexpr bool is_central_cube(const int idoubling) {
  return idoubling >= static_cast<int>(radial_flag::middle_central_cube) &&
         idoubling <= static_cast<int>(radial_flag::top_central_cube);
}

} // namespace specfem::globe
