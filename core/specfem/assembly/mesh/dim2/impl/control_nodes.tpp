#pragma once

#include "control_nodes.hpp"
#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"
#include "mesh_to_compute_mapping.hpp"
#include <Kokkos_Core.hpp>

specfem::assembly::mesh_impl::control_nodes<specfem::dimension::type::dim2>::
    control_nodes(
        const specfem::assembly::mesh_impl::mesh_to_compute_mapping<
            specfem::dimension::type::dim2> &mapping,
        const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
            &control_nodes)
    : ngnod(control_nodes.ngnod), nspec(control_nodes.nspec),
      control_node_mapping(
          "specfem::assembly::control_nodes::control_node_mapping",
          control_nodes.nspec, control_nodes.ngnod),
      control_node_coord("specfem::assembly::control_nodes::control_node_coord",
                         ndim, control_nodes.nspec, control_nodes.ngnod),
      h_control_node_mapping(Kokkos::create_mirror_view(control_node_mapping)),
      h_control_node_coord(Kokkos::create_mirror_view(control_node_coord)) {

  Kokkos::parallel_for(
      "specfem::assembly::control_nodes::assign_control_node_mapping",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Rank<2, Kokkos::Iterate::Right> >(
          { 0, 0 }, { ngnod, nspec }),
      [=](const int in, const int ispec) {
        const int ispec_mesh = mapping.compute_to_mesh(ispec);
        const int index = control_nodes.knods(in, ispec_mesh);
        h_control_node_mapping(ispec, in) = index;
        h_control_node_coord(0, ispec, in) = control_nodes.coord(0, index);
        h_control_node_coord(1, ispec, in) = control_nodes.coord(1, index);
      });

  Kokkos::deep_copy(control_node_mapping, h_control_node_mapping);
  Kokkos::deep_copy(control_node_coord, h_control_node_coord);

  return;
}
