#pragma once

#include "control_nodes.hpp"
#include "mesh/mesh.hpp"
#include "enumerations/dimension.hpp"
#include <Kokkos_Core.hpp>

specfem::assembly::mesh_impl::control_nodes<specfem::dimension::type::dim3>::
    control_nodes(
        const specfem::mesh::control_nodes<dimension_tag> &control_nodes)
    : nspec(control_nodes.nspec), ngnod(control_nodes.ngnod),
      control_node_coordinates("specfem::assembly::mesh::control_nodes",
                               control_nodes.nspec, control_nodes.ngnod, ndim),
      h_control_node_coordinates(
          Kokkos::create_mirror_view(control_node_coordinates)) {



  Kokkos::parallel_for(
      "specfem::assembly::mesh::control_nodes::copy_to_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Rank<2> >(
          { 0, 0 }, { control_nodes.nspec, control_nodes.ngnod }),
      [=](const int ispec, const int ia) {
        const int index = control_nodes.index_mapping(ispec, ia);
        for (int idim = 0; idim < ndim; ++idim)
          h_control_node_coordinates(ispec, ia, idim) =
              control_nodes.coordinates(index, idim);
      });

  Kokkos::fence();
  Kokkos::deep_copy(control_node_coordinates, h_control_node_coordinates);
  return;
}
