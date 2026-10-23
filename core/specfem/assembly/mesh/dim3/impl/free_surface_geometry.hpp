#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/mesh/mesh_base.hpp"
#include "specfem/mesh_entity.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {
template <specfem::element::dimension_tag Dimension> struct mesh;
} // namespace specfem::assembly

namespace specfem::assembly::mesh_impl {

/**
 * @brief Free-surface faces of an assembled 3D mesh, stored as geometry.
 *
 * The outer (free) surface used to place a depth-based coordinate against the
 * topographic surface (@ref specfem::algorithms::project_onto_surface). Unlike
 * the raw @ref specfem::mesh::acoustic_free_surface this is geometry, not a
 * boundary condition, and @ref element_index is already in compute-element
 * ordering so no mesh-to-compute remap is needed at use time. Default
 * constructs empty (@ref nfaces == 0), which the projection treats as a flat
 * surface.
 */
struct free_surface_geometry {
  /** @brief Number of free-surface faces. */
  int nfaces = 0;

  /** @brief Compute-element index for each free-surface face. */
  Kokkos::View<int *, Kokkos::HostSpace> element_index;

  /** @brief Local face identifier for each free-surface face. */
  Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace> face_type;

  free_surface_geometry() = default;

  /**
   * @brief Build from a regional mesh's acoustic free surface (Z_MAX faces).
   *
   * @param mesh Raw regional mesh carrying the acoustic free-surface faces
   * @param assembly_mesh Assembled mesh, for the mesh-to-compute remap
   */
  free_surface_geometry(
      const specfem::mesh::cartesian3d_mesh &mesh,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
          &assembly_mesh);

  /**
   * @brief Build from a globe mesh's geometric free surface.
   *
   * @param mesh Raw globe mesh carrying @c globe.free_surface
   * @param assembly_mesh Assembled mesh, for the mesh-to-compute remap
   */
  free_surface_geometry(
      const specfem::mesh::globe3d_mesh &mesh,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
          &assembly_mesh);
};

} // namespace specfem::assembly::mesh_impl
