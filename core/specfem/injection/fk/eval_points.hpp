#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {
namespace fk {

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
 * All Views are host-resident: the FK driver (grouping, spectra, IFFTs) runs on
 * the host, and only the final grouped @c FkResult is copied to device for
 * sampling. Optional Views (@c normal_*, @c lame_ratio_*, @c medium) have
 * extent 0 when the corresponding feature is not requested; the driver checks
 * @c extent(0) > 0 before dereferencing.
 *
 * @c z is depth, positive downward from the model top.
 */
struct EvalPoints {
  Kokkos::View<type_real *, Kokkos::HostSpace> x{}; ///< x-coordinates in m
  Kokkos::View<type_real *, Kokkos::HostSpace> y{}; ///< y-coordinates in m
  Kokkos::View<type_real *, Kokkos::HostSpace> z{}; ///< depth (positive down),
                                                    ///< m

  Kokkos::View<type_real *, Kokkos::HostSpace> normal_x{}; ///< x-component of
                                                           ///< outward unit
                                                           ///< normal (traction
                                                           ///< only; empty if
                                                           ///< unused)
  Kokkos::View<type_real *, Kokkos::HostSpace> normal_y{}; ///< y-component of
                                                           ///< outward unit
                                                           ///< normal (traction
                                                           ///< only; empty if
                                                           ///< unused)
  Kokkos::View<type_real *, Kokkos::HostSpace> normal_z{}; ///< z-component of
                                                           ///< outward unit
                                                           ///< normal (traction
                                                           ///< only; empty if
                                                           ///< unused)

  Kokkos::View<type_real *, Kokkos::HostSpace>
      lame_ratio_xi1{}; ///< First Lame-ratio factor \f$\xi_1\f$ at each point
                        ///< (traction only; empty if unused)
  Kokkos::View<type_real *, Kokkos::HostSpace>
      lame_ratio_xim{}; ///< Lame-ratio factor \f$\xi_\mu\f$ at each point
                        ///< (traction only; empty if unused)
  Kokkos::View<type_real *, Kokkos::HostSpace>
      lame_ratio_bulk{}; ///< Bulk Lame-ratio factor at each point
                         ///< (traction only; empty if unused)

  /** @brief Per-point medium selection. Size 0 means every point follows the
   *         layer table (@c point_medium::from_depth). A forced value selects
   *         the medium for a point sitting on the fluid-solid interface.
   */
  Kokkos::View<point_medium *, Kokkos::HostSpace> medium{};

  /**
   * @brief Return the number of evaluation points.
   * @return number of points (from the extent of @c x)
   */
  int size() const { return static_cast<int>(x.extent(0)); }
};

} // namespace fk
} // namespace injection
} // namespace specfem
