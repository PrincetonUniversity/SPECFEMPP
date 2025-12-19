#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief 3D control node data for assembly-optimized spectral elements.
 *
 * Stores control node coordinates and indices for 3D spectral elements
 * using Kokkos views for efficient device/host access.
 *
 * @see specfem::mesh::control_nodes
 */
template <> struct control_nodes<specfem::dimension::type::dim3> {
private:
  constexpr static int ndim = 3;

public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  /**
   * @brief Kokkos view type for 3D coordinate storage.
   *
   * Dimensions: [nspec, ngnod, 3] for (x, y, z) coordinates.
   */
  using ControlNodeCoordinatesView =
      Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for node index mapping.
   *
   * Dimensions: [nspec, ngnod] for element-to-node mapping.
   */
  using ControlNodeIndexView =
      Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  int nspec; ///< Number of spectral elements
  int ngnod; ///< Number of control nodes per element

  ControlNodeCoordinatesView control_node_coordinates; ///< Device coordinates
  ControlNodeCoordinatesView::HostMirror
      h_control_node_coordinates; ///< Host coordinates

  ControlNodeIndexView control_node_index;               ///< Device indices
  ControlNodeIndexView::HostMirror h_control_node_index; ///< Host indices

  /**
   * @brief Default constructor.
   */
  control_nodes() = default;

  /**
   * @brief Constructor from mesh control nodes.
   *
   * Copies and reorganizes mesh control node data for assembly operations.
   *
   * @param control_nodes Source mesh control nodes
   */
  control_nodes(
      const specfem::mesh::control_nodes<dimension_tag> &control_nodes);
};

} // namespace specfem::assembly::mesh_impl
