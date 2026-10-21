#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <string>

namespace specfem {
namespace injection {

/**
 * @brief Host list of the points at which the incident field is requested.
 *
 * One row per injection point, in the mesh frame (z up) and SI units. The
 * provider session converts it to the plugin ABI (@c spfp_points) when it sets
 * the plugin up.
 */
struct InjectionPoints {
  using view_type = Kokkos::View<type_real *, Kokkos::HostSpace>;

  view_type x;   ///< x-coordinates in m
  view_type y;   ///< y-coordinates in m
  view_type z;   ///< z-coordinates in m (up)
  view_type nx;  ///< Outward unit normal, x-component
  view_type ny;  ///< Outward unit normal, y-component
  view_type nz;  ///< Outward unit normal, z-component
  view_type rho; ///< Local density in kg/m^3
  view_type vp;  ///< Local P-wave speed in m/s
  view_type vs;  ///< Local S-wave speed in m/s (0 for fluids)
  Kokkos::View<std::int32_t *, Kokkos::HostSpace> medium; ///< Medium of each
                                                          ///< row (spfp_medium)

  InjectionPoints() = default;

  /**
   * @brief Allocate zero-initialized arrays for a number of rows.
   * @param nrows Number of injection points
   */
  explicit InjectionPoints(const std::int64_t nrows)
      : x("injection::points::x", nrows), y("injection::points::y", nrows),
        z("injection::points::z", nrows), nx("injection::points::nx", nrows),
        ny("injection::points::ny", nrows), nz("injection::points::nz", nrows),
        rho("injection::points::rho", nrows),
        vp("injection::points::vp", nrows), vs("injection::points::vs", nrows),
        medium("injection::points::medium", nrows) {}

  /**
   * @brief Number of rows.
   * @return Number of injection points
   */
  [[nodiscard]] std::int64_t size() const {
    return static_cast<std::int64_t>(x.extent(0));
  }
};

} // namespace injection
} // namespace specfem
