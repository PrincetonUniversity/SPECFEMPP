#pragma once

#include "specfem/assembly/coordinate_conversion.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geocentric_projection.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/transform.hpp"

#include <stdexcept>
#include <type_traits>

// Globe (Globe3D) mesh: composes the geographic -> geocentric -> Cartesian
// transforms on the sphere. Perfect-sphere case only; elliptical / topographic
// meshes throw rather than silently approximating.
template <typename Target>
specfem::assembly::coordinate_conversion_impl::resolved_t<Target>
specfem::assembly::coordinate_conversion_impl::convert_globe(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::globe3d_mesh &raw_mesh) {

  using cartesian = specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>;

  const auto &globe = raw_mesh.globe;
  const auto guard_unsupported = [&globe]() {

    if (globe.model_config.ellipticity || globe.model_config.topography)

      throw std::runtime_error(
          "elliptical/topographic globe coordinate resolution not yet "
          "implemented (issue #2058)");
  };

  if constexpr (std::is_same_v<Target, cartesian>) {

    const auto &surface = raw_mesh.boundaries.acoustic_free_surface;

    if (const auto *point = as<cartesian>(input))
      return resolve_cartesian(*point, mesh, surface);

    guard_unsupported();

    const double r_planet = globe.planet_constants.value().r_planet();

    if (const auto *geographic =
            as<specfem::coordinate_systems::geographic_coordinates>(input)) {
      const auto geocentric = specfem::coordinate_systems::transform<
          specfem::coordinate_systems::geocentric_coordinates>(
          *geographic,
          specfem::coordinate_systems::geocentric_projection_config{ r_planet });
      return resolve_cartesian(
          specfem::coordinate_systems::transform<cartesian>(geocentric), mesh,
          surface);
    }

    if (const auto *geocentric =
            as<specfem::coordinate_systems::geocentric_coordinates>(input))
      return resolve_cartesian(
          specfem::coordinate_systems::transform<cartesian>(*geocentric), mesh,
          surface);
    throw std::runtime_error("specfem::assembly::to: unknown coordinate type");

  } else if constexpr (std::is_same_v<
                           Target, specfem::coordinate_systems::
                                       geocentric_coordinates>) {
    (void)mesh;
    guard_unsupported();

    if (const auto *point = as<cartesian>(input))
      return specfem::coordinate_systems::transform<
          specfem::coordinate_systems::geocentric_coordinates>(*point);
    throw std::runtime_error(
        "specfem::assembly::to<geocentric>: expected a Cartesian input");

  } else if constexpr (std::is_same_v<
                           Target, specfem::coordinate_systems::
                                       geographic_coordinates>) {
    (void)mesh;
    guard_unsupported();
    const double r_planet = globe.planet_constants.value().r_planet();

    if (const auto *point = as<cartesian>(input)) {

      const auto geocentric = specfem::coordinate_systems::transform<
          specfem::coordinate_systems::geocentric_coordinates>(*point);

      return specfem::coordinate_systems::transform<
          specfem::coordinate_systems::geographic_coordinates>(
          geocentric,
          specfem::coordinate_systems::geocentric_projection_config{
              r_planet });
    }
    throw std::runtime_error(
        "specfem::assembly::to<geographic>: expected a Cartesian input");

  } else {
    static_assert(always_false_v<Target>,
                  "specfem::assembly::to: unsupported target coordinate "
                  "system");
  }
}
