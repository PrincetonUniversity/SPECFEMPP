#pragma once

#include "control_nodes.hpp"
#include "reference_points.hpp"
#include "specfem/mesh.hpp"

specfem::assembly::mesh_impl::reference_points<
    specfem::element::dimension_tag::dim3>::
    reference_points(
        const specfem::assembly::mesh_impl::points<dimension_tag>
            &final_points,
        const specfem::assembly::mesh_impl::mesh_to_compute_mapping<
            dimension_tag> &mapping,
        const specfem::assembly::mesh_impl::shape_functions<dimension_tag>
            &shape_functions,
        const specfem::mesh::control_nodes<dimension_tag> &control_nodes,
        const specfem::mesh::control_nodes<dimension_tag>::CoordinatesViewType
            &reference_anchor_coordinates)
    : has_reference_geometry(true) {
  // Shallow copy with the anchor coordinates swapped: the reference GLL
  // points must use the same element-to-anchor connectivity and reordering
  // as the final geometry.
  auto reference_control_nodes = control_nodes;
  reference_control_nodes.coordinates = reference_anchor_coordinates;
  const specfem::assembly::mesh_impl::control_nodes<dimension_tag>
      assembled_reference_nodes(mapping, reference_control_nodes);
  reference_gll_points = { final_points, assembled_reference_nodes,
                           shape_functions };
}
