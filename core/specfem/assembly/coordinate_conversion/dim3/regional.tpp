#pragma once

#include "specfem/assembly/coordinate_conversion.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/to.hpp"
#include "specfem/coordinate_systems/utm_projection.hpp"

#include <stdexcept>
#include <type_traits>

// Regional (Cartesian3D) mesh: geographic input is placed through the UTM
// projection, then (like any Cartesian input) against the free surface.
template <typename Target>
specfem::assembly::coordinate_conversion_impl::resolved_t<Target>
specfem::assembly::coordinate_conversion_impl::convert_regional(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::cartesian3d_mesh &raw_mesh) {

  using cartesian = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;
  const specfem::coordinate_systems::utm_projection_config utm{
    raw_mesh.utm_projection_zone
  };
  const auto &surface = raw_mesh.boundaries.acoustic_free_surface;

  if constexpr (std::is_same_v<Target, cartesian>) {

    if (const auto *point = as<cartesian>(input))
      return resolve_cartesian(*point, mesh, surface);

    if (raw_mesh.suppress_utm_projection)
      throw std::runtime_error("specfem::assembly::to: mesh has no projection "
                               "for non-Cartesian coordinates");
    if (const auto *geographic =
            as<specfem::coordinate_systems::geographic_coordinates>(input))
      return resolve_cartesian(
          specfem::coordinate_systems::to<cartesian>(*geographic, utm),
          mesh, surface);
    throw std::runtime_error("specfem::assembly::to: unknown coordinate type");

  } else if constexpr (std::is_same_v<
                           Target, specfem::coordinate_systems::
                                       geographic_coordinates>) {

    if (raw_mesh.suppress_utm_projection)
      throw std::runtime_error(
          "specfem::assembly::to<geographic>: mesh has no projection");

    if (const auto *point = as<cartesian>(input))
      return specfem::coordinate_systems::to<
          specfem::coordinate_systems::geographic_coordinates>(*point, utm);
    throw std::runtime_error(
        "specfem::assembly::to<geographic>: expected a Cartesian input");

  } else {
    static_assert(always_false_v<Target>,
                  "specfem::assembly::to: a regional mesh supports only "
                  "Cartesian and geographic targets");
  }
}
