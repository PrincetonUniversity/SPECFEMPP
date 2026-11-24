#pragma once

#include "specfem/plugin/injection_abi.h"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace specfem {
namespace injection {

/**
 * @brief A quantity requested from the plugin and the number of buffer
 * components it occupies.
 *
 * The component count depends on the medium (e.g. 3 for elastic velocity, 1
 * for acoustic potential rate) and is decided by SPECFEM++, not the plugin.
 */
struct QuantityRequest {
  specfem_quantity quantity = specfem_quantity_displacement; ///< Quantity
  int components = 0; ///< Buffer components it occupies
};

/**
 * @brief Injection points of one material system and the quantities
 * requested there.
 *
 * Covers rows [begin, begin + count) of specfem::injection::InjectionPoints.
 * Property names follow the SPECFEM++ point property container of the system
 * (e.g. @c kappa, @c mu, @c rho for elastic/isotropic).
 */
struct PointGroup {
  using properties_view =
      Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>;

  specfem_medium medium = specfem_medium_elastic;         ///< Medium
  specfem_property property = specfem_property_isotropic; ///< Parameterization
  std::int64_t begin = 0;                  ///< First row of the group
  std::int64_t count = 0;                  ///< Number of rows (may be 0)
  std::vector<std::string> property_names; ///< Material property names
  properties_view properties; ///< [count][property_names.size()] values
  std::vector<QuantityRequest> quantities; ///< Requested quantities, in slot
                                           ///< order

  /**
   * @brief Number of buffer components per point of the group.
   * @return Sum of the components of the requested quantities
   */
  [[nodiscard]] int components() const {
    return std::transform_reduce(
        quantities.begin(), quantities.end(), 0, std::plus<>{},
        [](const QuantityRequest &request) { return request.components; });
  }
};

/**
 * @brief Host list of the points at which the incident field is requested.
 *
 * One row per injection point, in the mesh frame (z up) and SI units. Rows are
 * ordered by group, and the groups partition [0, size()) in order. Every rank
 * holds the same groups, in the same order, including groups with no rows.
 * The provider session converts it to the plugin ABI (@c specfem_points) when
 * it sets the plugin up.
 */
struct InjectionPoints {
  using view_type = Kokkos::View<type_real *, Kokkos::HostSpace>;

  view_type x;                    ///< x-coordinates in m
  view_type y;                    ///< y-coordinates in m
  view_type z;                    ///< z-coordinates in m (up)
  view_type nx;                   ///< Outward unit normal, x-component
  view_type ny;                   ///< Outward unit normal, y-component
  view_type nz;                   ///< Outward unit normal, z-component
  std::vector<PointGroup> groups; ///< Groups partitioning the rows

  InjectionPoints() = default;

  /**
   * @brief Allocate zero-initialized coordinate arrays for a number of rows.
   * @param nrows Number of injection points
   */
  explicit InjectionPoints(const std::int64_t nrows)
      : x("injection::points::x", nrows), y("injection::points::y", nrows),
        z("injection::points::z", nrows), nx("injection::points::nx", nrows),
        ny("injection::points::ny", nrows), nz("injection::points::nz", nrows) {
  }

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
