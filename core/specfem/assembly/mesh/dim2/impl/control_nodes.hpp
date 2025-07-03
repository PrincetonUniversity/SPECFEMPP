#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "mesh_to_compute_mapping.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::impl {
/**
 * @brief Spectral element control nodes
 *
 */
template <> struct control_nodes<specfem::dimension::type::dim2> {
public:
  int nspec; ///< Number of spectral elements
  int ngnod; ///< Number of control nodes

  using IndexMappingViewType =
      Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  using CoordViewType = Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                     Kokkos::DefaultExecutionSpace>;

  IndexMappingViewType control_node_mapping; ///< Global index number for every
                                             ///< control node
  CoordViewType control_node_coord; ///< (x, z) for every distinct control
                                    ///< node
  IndexMappingViewType::HostMirror h_control_node_mapping; ///< Global element
                                                           ///< number for every
                                                           ///< control node
  CoordViewType::HostMirror h_control_node_coord;          ///< (x, z) for every
                                                           ///< distinct control
                                                           ///< node

  control_nodes(
      const specfem::assembly::impl::mesh_to_compute_mapping<
          specfem::dimension::type::dim2> &mapping,
      const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
          &control_nodes);

  control_nodes() = default;
};

} // namespace specfem::assembly::impl
