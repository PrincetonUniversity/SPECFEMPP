#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {

/**
 * @brief Method-agnostic points at which an incident field is requested.
 *
 * This is the entire contract between the boundary layer (which enumerates the
 * points and reads the local mesh material) and any injection method (which
 * decides how to synthesize displacement/traction there). It carries geometry,
 * medium, and the local isotropic material only — no method-specific quantities
 * (e.g. the FK Lamé ratios live inside the FK provider, derived from
 * @c vp / @c vs here).
 *
 * All Views are host-resident: provider preprocessing runs on the host.
 */
struct injection_points {
  Kokkos::View<type_real *, Kokkos::HostSpace> x{}; ///< x-coordinate in m
  Kokkos::View<type_real *, Kokkos::HostSpace> y{}; ///< y-coordinate in m
  Kokkos::View<type_real *, Kokkos::HostSpace> z{}; ///< depth (positive down),
                                                    ///< m

  Kokkos::View<type_real *, Kokkos::HostSpace> normal_x{}; ///< x-component of
                                                           ///< the outward unit
                                                           ///< normal
  Kokkos::View<type_real *, Kokkos::HostSpace> normal_y{}; ///< y-component of
                                                           ///< the outward unit
                                                           ///< normal
  Kokkos::View<type_real *, Kokkos::HostSpace> normal_z{}; ///< z-component of
                                                           ///< the outward unit
                                                           ///< normal

  /** @brief Medium of the mesh element the point belongs to. */
  Kokkos::View<specfem::element::medium_tag *, Kokkos::HostSpace> medium{};

  Kokkos::View<type_real *, Kokkos::HostSpace> rho{}; ///< Local density
  Kokkos::View<type_real *, Kokkos::HostSpace> vp{};  ///< Local P-wave speed
  Kokkos::View<type_real *, Kokkos::HostSpace> vs{};  ///< Local S-wave speed (0
                                                      ///< for acoustic points)

  /**
   * @brief Return the number of points.
   * @return number of points (from the extent of @c x)
   */
  int size() const { return static_cast<int>(x.extent(0)); }
};

} // namespace injection
} // namespace specfem
