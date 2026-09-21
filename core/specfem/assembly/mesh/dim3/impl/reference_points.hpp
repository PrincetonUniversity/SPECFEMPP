#pragma once

#include "mesh_to_compute_mapping.hpp"
#include "points.hpp"
#include "shape_functions.hpp"
#include "specfem/element.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief Reference (undeformed) GLL coordinates for model sampling.
 *
 * The globe mesher samples the physical model on the spherical,
 * Moho-stretched mesh — before topography and ellipticity move the points.
 * When the database provides the matching reference anchors, this mixin
 * stores their interpolation to the GLL points, computed with the same shape
 * functions and element ordering as the final geometry.
 *
 * When no reference anchors are given, the container stays empty and
 * consumers fall back to the final points (which are then identical by
 * definition).
 *
 * @see specfem::assembly::mesh_impl::points
 */
template <> struct reference_points<specfem::element::dimension_tag::dim3> {
public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

  bool has_reference_geometry = false; ///< True when reference anchors were
                                       ///< provided by the database

  /**
   * @brief GLL coordinates of the reference (undeformed) geometry.
   *
   * Shares its global numbering views with the final points; empty when
   * @ref has_reference_geometry is false.
   */
  specfem::assembly::mesh_impl::points<dimension_tag> reference_gll_points;

  /**
   * @brief Default constructor. No reference geometry.
   */
  reference_points() = default;

  /**
   * @brief Constructor interpolating reference anchors to GLL points.
   *
   * Reuses the assembly control-node reordering and the final geometry's
   * shape functions so the reference coordinates are interpolated exactly
   * like the final ones.
   *
   * @param final_points Final GLL point set whose numbering is shared
   * @param mapping Mesh-to-compute element reordering
   * @param shape_functions Shape function values at GLL points
   * @param control_nodes Raw mesh control nodes providing element-to-anchor
   * indices
   * @param reference_anchor_coordinates Reference anchor coordinates indexed
   * by global anchor node
   */
  reference_points(
      const specfem::assembly::mesh_impl::points<dimension_tag> &final_points,
      const specfem::assembly::mesh_impl::mesh_to_compute_mapping<dimension_tag>
          &mapping,
      const specfem::assembly::mesh_impl::shape_functions<dimension_tag>
          &shape_functions,
      const specfem::mesh::control_nodes<dimension_tag> &control_nodes,
      const specfem::mesh::control_nodes<dimension_tag>::CoordinatesViewType
          &reference_anchor_coordinates);
};

} // namespace specfem::assembly::mesh_impl
