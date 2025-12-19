#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "mesh_to_compute_mapping.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief 2D control node data structure optimized for computational assembly.
 *
 * The control nodes define the geometry of each spectral element and are used
 * for coordinate transformations and shape function evaluations.
 *
 * @tparam specfem::dimension::type::dim2 Template parameter specializing for
 * 2D.
 *
 * @see specfem::mesh::control_nodes
 */
template <> struct control_nodes<specfem::dimension::type::dim2> {
public:
  /**
   * @brief Number of spectral elements in the mesh.
   *
   */
  int nspec;

  /**
   * @brief Number of control nodes per spectral element.
   *
   */
  int ngnod;

  /**
   * @brief Kokkos view type for integer index mappings.
   *
   */
  using ControlNodeIndexView =
      Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for real-valued coordinate data.
   *
   */
  using ControlNodeCoordinatesView =
      Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Device view containing control node indices in compute ordering.
   *
   * Dimensions: [nspec, ngnod]. Maps from compute-ordered spectral elements
   * to global control node indices.
   */
  ControlNodeIndexView control_node_mapping;

  /**
   * @brief Device view containing control node coordinates.
   *
   * Dimensions: [ndim, nspec, ngnod] where ndim=2 for 2D problems.
   * Stores (x, z) coordinates for each control node of each spectral element.
   */
  ControlNodeCoordinatesView control_node_coord;

  /**
   * @brief Host mirror view of control node index mapping.
   *
   */
  ControlNodeIndexView::HostMirror h_control_node_mapping;

  /**
   * @brief Host mirror view of control node coordinates.
   *
   */
  ControlNodeCoordinatesView::HostMirror h_control_node_coord;

  /**
   * @brief Construct control nodes in from mesh control nodes.
   *
   * @param mapping Mapping object providing mesh-to-compute element reordering
   * @param control_nodes Original mesh control node data in mesh ordering
   *
   */
  control_nodes(
      const specfem::assembly::mesh_impl::mesh_to_compute_mapping<
          specfem::dimension::type::dim2> &mapping,
      const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
          &control_nodes);

  /**
   * @brief Default constructor.
   *
   */
  control_nodes() = default;
};

} // namespace specfem::assembly::mesh_impl
