#include "specfem/assembly/coordinate_resolver.hpp"

#include "specfem/algorithms/locate_point.hpp"
#include "specfem/setup.hpp"

#include <array>
#include <optional>

template <specfem::element::dimension_tag DimensionTag>
specfem::assembly::coordinate_resolver<DimensionTag>::Result
specfem::assembly::coordinate_resolver<DimensionTag>::resolve_cartesian(
    specfem::coordinate_systems::cartesian_coordinates<DimensionTag> &cartesian,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::mesh::acoustic_free_surface<DimensionTag> &surface) const {

  if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
    (void)mesh;
    (void)surface; // no topographic depth resolution in dim2

    if (!cartesian.origin.has_value())
      cartesian.origin = std::array<double, 2>{ 0.0, 0.0 };
    const auto &origin = *cartesian.origin;
    return { { static_cast<type_real>(cartesian.x + origin[0]),
               static_cast<type_real>(cartesian.z + origin[1]) },
             std::nullopt };
  } else {
    std::optional<type_real> topography;
    if (!cartesian.origin.has_value()) {
      // Depth-based: set the origin elevation from the topographic surface
      // above (x, y). With no free surface the projection returns z = 0 (flat).
      const auto landing = specfem::algorithms::project_onto_surface(
          mesh, surface,
          { static_cast<type_real>(cartesian.x),
            static_cast<type_real>(cartesian.y),
            static_cast<type_real>(cartesian.z) });
      cartesian.origin =
          std::array<double, 3>{ 0.0, 0.0, static_cast<double>(landing.z) };
      topography = landing.z;
    }
    const auto &origin = *cartesian.origin;
    return { { static_cast<type_real>(cartesian.x + origin[0]),
               static_cast<type_real>(cartesian.y + origin[1]),
               static_cast<type_real>(cartesian.z + origin[2]) },
             topography };
  }
}

template class specfem::assembly::coordinate_resolver<
    specfem::element::dimension_tag::dim2>;
template class specfem::assembly::coordinate_resolver<
    specfem::element::dimension_tag::dim3>;
