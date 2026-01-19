#pragma once

#include "control_nodes.hpp"
#include "enumerations/interface.hpp"
#include "shape_functions.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief 3D quadrature point coordinates and global indexing.
 *
 * Stores coordinates and global indices for all GLL quadrature points
 * in 3D spectral elements using Kokkos views for device/host access.
 *
 * @see specfem::assembly::mesh_impl::control_nodes,
 * specfem::assembly::mesh_impl::shape_functions
 */
template <> struct points<specfem::dimension::type::dim3> {
public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  /**
   * @brief Index mapping view type.
   *
   * Dimensions: [nspec, ngllz, nglly, ngllx] for local-to-global mapping.
   */
  using IndexMappingViewType =
      Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Coordinate view type.
   *
   * Dimensions: [ndim, nspec, ngllz, nglly, ngllx] for (x, y, z) coordinates.
   */
  using CoordViewType = Kokkos::View<type_real *****, Kokkos::LayoutLeft,
                                     Kokkos::DefaultExecutionSpace>;

private:
  constexpr static int ndim = 3;

public:
  IndexMappingViewType index_mapping;               ///< Device index mapping
  IndexMappingViewType::HostMirror h_index_mapping; ///< Host index mapping

  CoordViewType coord;               ///< Device coordinates
  CoordViewType::HostMirror h_coord; ///< Host coordinates

  int nspec; ///< Number of spectral elements
  int ngllz; ///< Number of GLL points in z dimension
  int nglly; ///< Number of GLL points in y dimension
  int ngllx; ///< Number of GLL points in x dimension

  int nglob; ///< Total number of global points

  /**
   * @brief Default constructor.
   */
  points() = default;

  /**
   * @brief Constructor computing coordinates from mesh components.
   *
   * Computes physical coordinates for all quadrature points using
   * control nodes and shape functions.
   *
   * @param nspec Number of spectral elements
   * @param ngllz Number of GLL points in z direction
   * @param nglly Number of GLL points in y direction
   * @param ngllx Number of GLL points in x direction
   * @param adjacency_graph Element adjacency information
   * @param control_nodes Element control node data
   * @param shape_functions Shape function values at GLL points
   */
  points(const int &nspec, const int &ngllz, const int &nglly, const int &ngllx,
         const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
         const specfem::assembly::mesh_impl::control_nodes<dimension_tag>
             &control_nodes,
         const specfem::assembly::mesh_impl::shape_functions<dimension_tag>
             &shape_functions);
};

} // namespace specfem::assembly::mesh_impl
