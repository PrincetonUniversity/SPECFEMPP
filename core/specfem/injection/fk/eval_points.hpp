#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {
namespace fk {

/**
 * @brief Evaluation points at which the FK solver computes displacement,
 *        traction, and pressure.
 *
 * All coordinate and material-factor Views live on the default Kokkos memory
 * space so that Kernel 2 can read them on-device without an extra deep_copy.
 * Views are constructed from a host mirror + @c Kokkos::deep_copy in the
 * caller; the Views here are the device-resident copies.
 *
 * Optional Views (@c normal_*, @c lame_ratio_*, @c elastic_override,
 * @c layer_override) have extent 0 when the corresponding feature is not
 * requested.  The driver checks @c extent(0) > 0 before dereferencing.
 */
struct EvalPoints {
  Kokkos::View<type_real *> x{}; ///< x-coordinates of evaluation points in m
  Kokkos::View<type_real *> y{}; ///< y-coordinates of evaluation points in m
  Kokkos::View<type_real *> z{}; ///< z-coordinates of evaluation points in m

  Kokkos::View<type_real *> normal_x{}; ///< x-component of outward unit normal
                                        ///< (traction only; empty if unused)
  Kokkos::View<type_real *> normal_y{}; ///< y-component of outward unit normal
                                        ///< (traction only; empty if unused)
  Kokkos::View<type_real *> normal_z{}; ///< z-component of outward unit normal
                                        ///< (traction only; empty if unused)

  Kokkos::View<type_real *> lame_ratio_xi1{};  ///< First Lame-ratio factor
                                               ///< \f$\xi_1\f$ at each point
                                               ///< (traction only; empty if
                                               ///< unused)
  Kokkos::View<type_real *> lame_ratio_xim{};  ///< Lame-ratio factor
                                               ///< \f$\xi_\mu\f$ at each point
                                               ///< (traction only; empty if
                                               ///< unused)
  Kokkos::View<type_real *> lame_ratio_bulk{}; ///< Bulk Lame-ratio factor at
                                               ///< each point (traction only;
                                               ///< empty if unused)

  /** @brief Per-point medium override: 'E' = elastic, 'A' = acoustic, 0 = use
   *         layer table.  Size 0 when unused (all points follow layer table).
   */
  Kokkos::View<char *> elastic_override{};

  /** @brief Per-point layer-index override (0-based).  Negative value or size 0
   *         means no override; the depth-lookup from @c LayeredModel is used.
   */
  Kokkos::View<int *> layer_override{};

  /**
   * @brief Return the number of evaluation points.
   * @return number of points (from the extent of @c x)
   */
  int size() const { return static_cast<int>(x.extent(0)); }
};

} // namespace fk
} // namespace injection
} // namespace specfem
