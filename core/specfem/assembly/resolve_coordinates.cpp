#include "specfem/assembly/resolve_coordinates.hpp"

#include "specfem/coordinate_systems/coordinates/cartesian_2d.hpp"
#include "specfem/coordinate_systems/coordinates/cartesian_3d.hpp"
#include "specfem/coordinate_systems/coordinates/cartesian_with_depth_3d.hpp"
#include "specfem/coordinate_systems/coordinates/geographic_3d.hpp"
#include "specfem/setup.hpp"

#include <stdexcept>

namespace {
constexpr auto dim2 = specfem::element::dimension_tag::dim2;
constexpr auto dim3 = specfem::element::dimension_tag::dim3;
} // namespace

// ── dim2 specialization ─────────────────────────────────────────────────────

template <>
specfem::point::global_coordinates<dim2>
specfem::assembly::resolve_coordinates<dim2>(
    const specfem::coordinate_systems::coordinates<dim2> &coords,
    const specfem::assembly::mesh<dim2> &mesh) {

  using namespace specfem::coordinate_systems;

  if (const auto *c = dynamic_cast<const cartesian_2d *>(&coords)) {
    return { static_cast<type_real>(c->x), static_cast<type_real>(c->z) };
  }

  throw std::runtime_error(
      "resolve_coordinates<dim2>: unknown coordinate type");
}

// ── dim3 specialization ─────────────────────────────────────────────────────

template <>
specfem::point::global_coordinates<dim3>
specfem::assembly::resolve_coordinates<dim3>(
    const specfem::coordinate_systems::coordinates<dim3> &coords,
    const specfem::assembly::mesh<dim3> &mesh) {

  using namespace specfem::coordinate_systems;

  if (const auto *c = dynamic_cast<const cartesian_3d *>(&coords)) {
    return { static_cast<type_real>(c->data.x),
             static_cast<type_real>(c->data.y),
             static_cast<type_real>(c->data.z) };
  }

  if (const auto *c = dynamic_cast<const cartesian_with_depth_3d *>(&coords)) {
    // Placeholder: z = -depth
    // When topography is available: z = topo_elevation_at(x, y, mesh) - depth
    return { static_cast<type_real>(c->x), static_cast<type_real>(c->y),
             static_cast<type_real>(-c->depth) };
  }

  if (const auto *c = dynamic_cast<const geographic_3d *>(&coords)) {
    // Geographic coordinate resolution requires UTM projection config
    // and topography — not yet implemented.
    throw std::runtime_error(
        "resolve_coordinates<dim3>: geographic coordinate resolution "
        "not yet implemented");
  }

  throw std::runtime_error(
      "resolve_coordinates<dim3>: unknown coordinate type");
}
