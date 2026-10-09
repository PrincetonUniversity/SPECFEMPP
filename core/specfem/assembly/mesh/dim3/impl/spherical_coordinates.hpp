#pragma once

#include "points.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"

namespace specfem::assembly::mesh_impl {

/**
 * @brief Setup-only spherical coordinates of the final, deformed GLL points.
 *
 * Host storage in compute-element order [ispec, iz, iy, ix, component], with
 * components (radius in metres, colatitude, longitude in radians). Double
 * precision preserves Cartesian round trips even when type_real is float.
 * The database has already applied PlanetConstants::r_planet() to coordinates;
 * neither radius normalization nor further ellipticity corrections belong here.
 * Model sampling must continue to use the separate reference coordinates.
 *
 * @note Globe's generated crust_mantle_impl_kernel_forward accepts d_rstore
 * but never reads it. Gravity and tensor rotations are precomputed at setup.
 * No current consumer needs a device mirror, so none is allocated. The assembly
 * constructor releases this cache after setup; consumers must not retain copies
 * of its view if the allocation is to be freed at that point.
 */
struct SphericalCoordinates {
  using HostView =
      Kokkos::View<double *****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  HostView h_coord; ///< Host (r, theta, phi); empty for Cartesian assemblies.

  SphericalCoordinates() = default;

  /**
   * @brief Build from final host coordinates, never reference coordinates.
   * @param points Assembled GLL points with current host coordinates in metres.
   */
  explicit SphericalCoordinates(
      const points<specfem::element::dimension_tag::dim3> &points)
      : h_coord("specfem::assembly::mesh::spherical_coordinates",
                points.h_coord.extent(0), points.h_coord.extent(1),
                points.h_coord.extent(2), points.h_coord.extent(3), 3) {
    for (std::size_t ispec = 0; ispec < h_coord.extent(0); ++ispec) {
      for (std::size_t iz = 0; iz < h_coord.extent(1); ++iz) {
        for (std::size_t iy = 0; iy < h_coord.extent(2); ++iy) {
          for (std::size_t ix = 0; ix < h_coord.extent(3); ++ix) {
            const auto spherical =
                specfem::coordinate_systems::geocentric_coordinates::
                    from_cartesian(points.h_coord(ispec, iz, iy, ix, 0),
                                   points.h_coord(ispec, iz, iy, ix, 1),
                                   points.h_coord(ispec, iz, iy, ix, 2));
            h_coord(ispec, iz, iy, ix, 0) = spherical.r;
            h_coord(ispec, iz, iy, ix, 1) = spherical.theta;
            h_coord(ispec, iz, iy, ix, 2) = spherical.phi;
          }
        }
      }
    }
  }

  /** @brief Drop the owning view after the last setup consumer. */
  void release() { h_coord = HostView{}; }
};

} // namespace specfem::assembly::mesh_impl
