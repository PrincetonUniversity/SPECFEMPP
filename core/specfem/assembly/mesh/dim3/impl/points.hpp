#pragma once

#include "control_nodes.hpp"
#include "shape_functions.hpp"
#include "specfem/data_access.hpp"
#include "specfem/element.hpp"
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
template <> struct points<specfem::element::dimension_tag::dim3> {
public:
  constexpr static auto data_class =
      specfem::data_access::DataClassType::global_coordinates; ///< Data class
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;

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
  constexpr static int ndim = specfem::element::dimension<
      specfem::element::dimension_tag::dim3>::dim; ///< Number of dimensions

public:
  IndexMappingViewType index_mapping; ///< Device index mapping
  IndexMappingViewType::host_mirror_type h_index_mapping; ///< Host index
                                                          ///< mapping

  CoordViewType coord;                     ///< Device coordinates
  CoordViewType::host_mirror_type h_coord; ///< Host coordinates

  /**
   * @brief Reference (undeformed) coordinates for model sampling.
   *
   * Interpolated from the database's reference (spherical + Moho-stretched)
   * anchors with the same shape functions and element ordering as @ref coord.
   * Aliases @ref coord when no reference geometry is given — the two
   * geometries are then identical by definition.
   *
   * @warning Model sampling only. Never use these coordinates for the
   * Jacobian, the mass matrix, or anything geometric — they do not describe
   * the deformed mesh the solver runs on.
   */
  CoordViewType reference_coord;
  CoordViewType::host_mirror_type h_reference_coord; ///< Host reference
                                                     ///< coordinates (see
                                                     ///< @ref reference_coord)

  type_real xmin; ///< Minimum x coordinate (for tolerance calculations)
  type_real xmax; ///< Maximum x coordinate (for tolerance calculations)
  type_real ymin; ///< Minimum y coordinate (for tolerance calculations)
  type_real ymax; ///< Maximum y coordinate (for tolerance calculations)
  type_real zmin; ///< Minimum z coordinate (for tolerance calculations)
  type_real zmax; ///< Maximum z coordinate (for tolerance calculations)

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
   * @param reference_control_nodes Optional assembled control nodes carrying
   * the reference (undeformed) anchor coordinates. When non-empty, @ref
   * reference_coord is contracted from them with the same shape functions and
   * ordering as @ref coord; when empty, @ref reference_coord aliases @ref
   * coord.
   */
  points(const int &nspec, const int &ngllz, const int &nglly, const int &ngllx,
         const Kokkos::View<specfem::element::medium_tag *, Kokkos::HostSpace>
             &medium_tags,
         const specfem::assembly::mesh_impl::adjacency_graph<dimension_tag>
             &adjacency_graph,
         const specfem::assembly::mesh_impl::control_nodes<dimension_tag>
             &control_nodes,
         const specfem::assembly::mesh_impl::shape_functions<dimension_tag>
             &shape_functions,
         const specfem::assembly::mesh_impl::control_nodes<dimension_tag>
             &reference_control_nodes = {});

private:
  /**
   * @brief Compute coordinate bounds over all quadrature points.
   *
   * Fills the min/max members from the device coordinate view and reduces
   * them across MPI ranks.
   */
  void compute_coordinate_bounds();
};

} // namespace specfem::assembly::mesh_impl
