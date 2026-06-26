#include "specfem/assembly/resolve_coordinates.hpp"

#include "specfem/algorithms/locate_point.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/utm.hpp"
#include "specfem/setup.hpp"

#include <stdexcept>

// ── dim2 specialization ─────────────────────────────────────────────────────

template <>
specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
specfem::assembly::resolve_coordinates<specfem::element::dimension_tag::dim2>(
    specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim2> &coords,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::mesh::acoustic_free_surface<
        specfem::element::dimension_tag::dim2> &surface,
    const std::optional<specfem::coordinate_systems::utm_projection_config>
        &utm_config) {

  (void)surface; // no topographic depth resolution in dim2

  using dim2 = std::integral_constant<specfem::element::dimension_tag,
                                      specfem::element::dimension_tag::dim2>;

  if (auto *c = dynamic_cast<specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim2> *>(&coords)) {
    if (!c->origin.has_value()) {
      // Flat topography fallback
      c->origin = { 0.0, 0.0 };
    }
    const auto &o = *c->origin;
    return { static_cast<type_real>(c->x + o[0]),
             static_cast<type_real>(c->z + o[1]) };
  }

  throw std::runtime_error(
      "resolve_coordinates<dim2>: unknown coordinate type");
}

// ── dim3 specialization ─────────────────────────────────────────────────────

template <>
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
specfem::assembly::resolve_coordinates<specfem::element::dimension_tag::dim3>(
    specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &coords,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::acoustic_free_surface<
        specfem::element::dimension_tag::dim3> &surface,
    const std::optional<specfem::coordinate_systems::utm_projection_config>
        &utm_config) {

  // Cartesian coordinates (absolute xyz or depth-based with origin)
  if (auto *c = dynamic_cast<specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim3> *>(&coords)) {
    if (!c->origin.has_value()) {
      // Depth-based: set the origin elevation from the topographic surface
      // above (x, y). With no free surface the projection returns z = 0 (flat).
      const auto landing = specfem::algorithms::project_onto_surface(
          mesh, surface,
          { static_cast<type_real>(c->x), static_cast<type_real>(c->y),
            static_cast<type_real>(c->z) });
      c->origin = { 0.0, 0.0, static_cast<double>(landing.z) };
    }
    const auto &o = *c->origin;
    return { static_cast<type_real>(c->x + o[0]),
             static_cast<type_real>(c->y + o[1]),
             static_cast<type_real>(c->z + o[2]) };
  }

  // Geographic coordinates — project via UTM, then resolve depth
  if (auto *c =
          dynamic_cast<specfem::coordinate_systems::geographic_coordinates *>(
              &coords)) {
    if (!utm_config.has_value()) {
      throw std::runtime_error(
          "resolve_coordinates<dim3>: geographic coordinates require "
          "utm_config for UTM projection");
    }

    // Forward-project lon/lat to UTM easting/northing.
    // The transform returns cartesian with origin=nullopt and z=-depth.
    auto cart = specfem::coordinate_systems::transform<
        specfem::coordinate_systems::cartesian_coordinates<
            specfem::element::dimension_tag::dim3>>(*c, *utm_config);

    // Resolve the projected cartesian coordinates (handles origin/depth)
    return specfem::assembly::resolve_coordinates(cart, mesh, surface,
                                                  utm_config);
  }

  // Geocentric coordinates — not yet implemented (Globe3D future)
  if (dynamic_cast<specfem::coordinate_systems::geocentric_coordinates *>(
          &coords)) {
    throw std::runtime_error(
        "resolve_coordinates<dim3>: geocentric coordinate resolution "
        "not yet implemented (Globe3D)");
  }

  throw std::runtime_error(
      "resolve_coordinates<dim3>: unknown coordinate type");
}
