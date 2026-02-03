#pragma once

#include "control_nodes.hpp"
#include "enumerations/dimension.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>

void initialize_control_nodes(
    specfem::assembly::mesh_impl::control_nodes<specfem::dimension::type::dim3>
        &dest,
    const specfem::mesh::control_nodes<specfem::dimension::type::dim3>
        &src) {

  // We use the device by default for better performance.

  using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;

  const auto coordinates =
      Kokkos::create_mirror_view_and_copy(MemorySpace(), src.coordinates);
  const auto control_node_index = Kokkos::create_mirror_view_and_copy(
      MemorySpace(), src.control_node_index);

  Kokkos::parallel_for(
      "specfem::assembly::mesh::control_nodes::copy_to_device",
      Kokkos::MDRangePolicy<Kokkos::Rank<2> >({ 0, 0 },
                                              { src.nspec, src.ngnod }),
      KOKKOS_LAMBDA(const int ispec, const int ia) {
        const int index = control_node_index(ispec, ia);
        dest.control_node_index(ispec, ia) = index;
        dest.control_node_coordinates(ispec, ia, 0) = coordinates(index, 0);
        dest.control_node_coordinates(ispec, ia, 1) = coordinates(index, 1);
        dest.control_node_coordinates(ispec, ia, 2) = coordinates(index, 2);
      });

  Kokkos::fence();

  Kokkos::deep_copy(dest.h_control_node_coordinates,
                    dest.control_node_coordinates);
  Kokkos::deep_copy(dest.h_control_node_index, dest.control_node_index);
  return;
}

specfem::assembly::mesh_impl::control_nodes<specfem::dimension::type::dim3>::
    control_nodes(const specfem::mesh::control_nodes<specfem::dimension::type::dim3>
                      &control_nodes)
    : nspec(control_nodes.nspec), ngnod(control_nodes.ngnod),
      control_node_index("specfem::assembly::mesh::control_nodes::index",
                         control_nodes.nspec, control_nodes.ngnod),
      h_control_node_index(Kokkos::create_mirror_view(control_node_index)),
      control_node_coordinates("specfem::assembly::mesh::control_nodes",
                               control_nodes.nspec, control_nodes.ngnod, ndim),
      h_control_node_coordinates(
          Kokkos::create_mirror_view(control_node_coordinates)) {

  initialize_control_nodes(*this, control_nodes);
}
