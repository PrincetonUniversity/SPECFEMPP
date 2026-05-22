#include "specfem/assembly/resolve_coordinates.hpp"

#include "specfem/coordinate_systems/input_coordinates/input_cartesian_2d.hpp"
#include "specfem/coordinate_systems/input_coordinates/input_cartesian_3d.hpp"
#include "specfem/coordinate_systems/input_coordinates/input_cartesian_with_depth_3d.hpp"
#include "specfem/coordinate_systems/input_coordinates/input_geographic_3d.hpp"
#include "specfem/setup.hpp"

#include <stdexcept>

// ── dim2 specialization ─────────────────────────────────────────────────────

template <>
specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
specfem::assembly::resolve_coordinates<specfem::element::dimension_tag::dim2>(
    const specfem::coordinate_systems::input_coordinates<
        specfem::element::dimension_tag::dim2> &coords,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2>
        &mesh) {

  using namespace specfem::coordinate_systems;

  if (const auto *c = dynamic_cast<const input_cartesian_2d *>(&coords)) {
    return { static_cast<type_real>(c->x), static_cast<type_real>(c->z) };
  }

  throw std::runtime_error(
      "resolve_coordinates<dim2>: unknown coordinate type");
}

// ── dim3 specialization ─────────────────────────────────────────────────────

template <>
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
specfem::assembly::resolve_coordinates<specfem::element::dimension_tag::dim3>(
    const specfem::coordinate_systems::input_coordinates<
        specfem::element::dimension_tag::dim3> &coords,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  using namespace specfem::coordinate_systems;

  if (const auto *c = dynamic_cast<const input_cartesian_3d *>(&coords)) {
    return { static_cast<type_real>(c->data.x),
             static_cast<type_real>(c->data.y),
             static_cast<type_real>(c->data.z) };
  }

  if (const auto *c =
          dynamic_cast<const input_cartesian_with_depth_3d *>(&coords)) {
    // TODO(#1867): This is a placeholder. Depth is measured positive downward
    // from the topographic surface. The correct conversion requires querying
    // the mesh for surface elevation at (x, y):
    //   z = topo_elevation_at(x, y, mesh) - depth
    // For now, assumes flat topography at z=0.
    return { static_cast<type_real>(c->x), static_cast<type_real>(c->y),
             static_cast<type_real>(-c->depth) };
  }

  if (const auto *c = dynamic_cast<const input_geographic_3d *>(&coords)) {
    // Geographic coordinate resolution requires UTM projection config
    // and topography — not yet implemented.
    throw std::runtime_error(
        "resolve_coordinates<dim3>: geographic coordinate resolution "
        "not yet implemented");
  }

  throw std::runtime_error(
      "resolve_coordinates<dim3>: unknown coordinate type");
}
