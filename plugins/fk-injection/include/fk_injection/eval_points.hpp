#pragma once

#include "fk_injection/config.hpp"
#include <vector>

namespace fk_injection {

/**
 * @brief Which medium a point's field is recovered in.
 *
 * from_depth resolves by the layer table; elastic / acoustic force the medium
 * (used when a point sits exactly on the fluid-solid interface).
 */
enum class point_medium { from_depth, elastic, acoustic };

/**
 * @brief Evaluation points at which the FK solver computes displacement,
 *        traction, and pressure.
 *
 * Optional members (@c normal_*, @c lame_ratio_*, @c medium) are empty when the
 * corresponding feature is not requested; the driver checks that they are not
 * empty before dereferencing.
 *
 * @c z is depth, positive downward from the model top.
 */
struct EvalPoints {
  std::vector<type_real> x{}; ///< x-coordinates in m
  std::vector<type_real> y{}; ///< y-coordinates in m
  std::vector<type_real> z{}; ///< depth (positive down), m

  std::vector<type_real> normal_x{}; ///< x-component of outward unit normal
                                     ///< (traction only; empty if unused)
  std::vector<type_real> normal_y{}; ///< y-component of outward unit normal
                                     ///< (traction only; empty if unused)
  std::vector<type_real> normal_z{}; ///< z-component of outward unit normal
                                     ///< (traction only; empty if unused)

  std::vector<type_real> lame_ratio_xi1{}; ///< First Lame-ratio factor
                                           ///< \f$\xi_1\f$ at each point
                                           ///< (traction only; empty if unused)
  std::vector<type_real> lame_ratio_xim{}; ///< Lame-ratio factor \f$\xi_\mu\f$
                                           ///< at each point (traction only;
                                           ///< empty if unused)
  std::vector<type_real> lame_ratio_bulk{}; ///< Bulk Lame-ratio factor at each
                                            ///< point (traction only; empty if
                                            ///< unused)

  /** @brief Per-point medium selection. Size 0 means every point follows the
   *         layer table (@c point_medium::from_depth). A forced value selects
   *         the medium for a point sitting on the fluid-solid interface.
   */
  std::vector<point_medium> medium{};

  /**
   * @brief Return the number of evaluation points.
   * @return number of points (from the size of @c x)
   */
  int size() const { return static_cast<int>(x.size()); }
};

} // namespace fk_injection
