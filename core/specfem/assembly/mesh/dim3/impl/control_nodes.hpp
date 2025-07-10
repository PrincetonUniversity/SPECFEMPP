#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

template <> struct control_nodes<specfem::dimension::type::dim3> {
private:
  constexpr static int ndim = 3;

public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;
  using ControlNodeCoordinatesView =
      Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>;

  int nspec;
  int ngnod;

  ControlNodeCoordinatesView control_node_coordinates;
  ControlNodeCoordinatesView::HostMirror h_control_node_coordinates;

  control_nodes() = default;

  control_nodes(
      const specfem::mesh::control_nodes<dimension_tag> &control_nodes);
};

} // namespace specfem::assembly::mesh_impl
