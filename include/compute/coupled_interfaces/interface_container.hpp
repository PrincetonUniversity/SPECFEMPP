#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/coupled_interfaces/interface_container.hpp"
#include "compute/element_types/element_types.hpp"
#include "edge/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace compute {

/**
 * @brief Information about coupled interfaces between MediumTag1 and MediumTag2
 *
 * @tparam MediumTag1 Self medium of the interface
 * @tparam MediumTag2 Other medium of the interface
 */
template <specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2>
struct interface_container {

private:
  using IndexView =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< indices
  using EdgeTypeView =
      Kokkos::View<specfem::enums::edge::type *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store edge types
  using EdgeFactorView =
      Kokkos::View<type_real **, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< integration factors
  using EdgeNormalView =
      Kokkos::View<type_real ***, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store edge normals

public:
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static specfem::element::medium_tag medium1_type =
      MediumTag1; ///< Self medium of the interface
  constexpr static specfem::element::medium_tag medium2_type =
      MediumTag2; ///< Other medium of the interface
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  interface_container() = default;

  interface_container(const int num_interfaces, const int ngll);

  /**
   * @brief Compute the interface for a given mesh
   *
   * @param mesh Finite element mesh information
   * @param points Assembly information
   * @param quadrature Quadrature information
   * @param partial_derivatives Partial derivatives for every quadrature point
   * @param properties Material properties for every quadrature point
   * @param mapping Mapping between mesh and compute spectral element indexing
   */
  interface_container(
      const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
      const specfem::compute::points &points,
      const specfem::compute::quadrature &quadrature,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::element_types &element_types,
      const specfem::compute::mesh_to_compute_mapping &mapping);

  /**
   * @brief Construct interface container from another container where mediums
   * are swapped
   *
   * @param other Interface container with swapped mediums
   */
  interface_container(const interface_container<MediumTag2, MediumTag1> &other)
      : num_interfaces(other.num_interfaces), num_points(other.num_points),
        medium1_index_mapping(other.medium2_index_mapping),
        h_medium1_index_mapping(other.h_medium2_index_mapping),
        medium2_index_mapping(other.medium1_index_mapping),
        h_medium2_index_mapping(other.h_medium1_index_mapping),
        medium1_edge_type(other.medium2_edge_type),
        h_medium1_edge_type(other.h_medium2_edge_type),
        medium2_edge_type(other.medium1_edge_type),
        h_medium2_edge_type(other.h_medium1_edge_type),
        medium1_edge_factor(other.medium2_edge_factor),
        h_medium1_edge_factor(other.h_medium2_edge_factor),
        medium2_edge_factor(other.medium1_edge_factor),
        h_medium2_edge_factor(other.h_medium1_edge_factor),
        medium1_edge_normal(other.medium2_edge_normal),
        h_medium1_edge_normal(other.h_medium2_edge_normal),
        medium2_edge_normal(other.medium1_edge_normal),
        h_medium2_edge_normal(other.h_medium1_edge_normal) {
    return;
  }
  ///@}

  int num_interfaces;              ///< Number of edges between the two mediums
  int num_points;                  ///< Number of points on the edges
  IndexView medium1_index_mapping; ///< Spectral element index for every edge on
                                   ///< self medium
  IndexView medium2_index_mapping; ///< Spectral element index for every edge on
                                   ///< other medium

  EdgeTypeView medium1_edge_type; ///< Edge orientation for every edge on self
                                  ///< medium
  EdgeTypeView medium2_edge_type; ///< Edge orientation for every edge on other
                                  ///< medium

  IndexView::HostMirror h_medium1_index_mapping; ///< Host mirror for @ref
                                                 ///< medium1_index_mapping
  IndexView::HostMirror h_medium2_index_mapping; ///< Host mirror for @ref
                                                 ///< medium2_index_mapping

  EdgeTypeView::HostMirror h_medium1_edge_type; ///< Host mirror for @ref
                                                ///< medium1_edge_type
  EdgeTypeView::HostMirror h_medium2_edge_type; ///< Host mirror for @ref
                                                ///< medium2_edge_type

  EdgeFactorView medium1_edge_factor; ///< Integration factors for every
                                      ///< quadrature point on self medium
  EdgeFactorView medium2_edge_factor; ///< Integration factors for every
                                      ///< quadrature point on other medium

  EdgeFactorView::HostMirror h_medium1_edge_factor; ///< Host mirror for @ref
                                                    ///< medium1_edge_factor
  EdgeFactorView::HostMirror h_medium2_edge_factor; ///< Host mirror for @ref
                                                    ///< medium2_edge_factor

  EdgeNormalView medium1_edge_normal; ///< Edge normals for every quadrature
                                      ///< point on self medium
  EdgeNormalView medium2_edge_normal; ///< Edge normals for every quadrature
                                      ///< point on other medium

  EdgeNormalView::HostMirror h_medium1_edge_normal; ///< Host mirror for @ref
                                                    ///< medium1_edge_normal
  EdgeNormalView::HostMirror h_medium2_edge_normal; ///< Host mirror for @ref
                                                    ///< medium2_edge_normal

  /**
   * @brief Get the spectral element index for elements on edges of the
   * interface
   *
   * @return std::tuple<IndexView, IndexView> Tuple containing the spectral
   * element indices. The first element is the indices for self medium and the
   * second element is the indices for the other medium
   */
  std::tuple<IndexView, IndexView> get_index_mapping() const {
    return std::make_tuple(medium1_index_mapping, medium2_index_mapping);
  }

  /**
   * @brief Get the orientation for the edges of the interface
   *
   * @return std::tuple<EdgeTypeView, EdgeTypeView> Tuple containing the edge
   * orientation. The first element is the edge orientation for self medium and
   * the second element is the edge orientation for the other medium
   */
  std::tuple<EdgeTypeView, EdgeTypeView> get_edge_type() const {
    return std::make_tuple(medium1_edge_type, medium2_edge_type);
  }

  /**
   * @brief Get the integration weights for edges on interface when computing
   * intergrals over self medium
   *
   * @return EdgeFactorView Integration weights for every quadrature point on
   * self medium
   */
  EdgeFactorView get_edge_factor() const { return medium1_edge_factor; }

  /**
   * @brief Get the normals to the edge at every quadrature point on the
   * interface on self medium
   *
   * @return EdgeNormalView Normals to the edge at every quadrature point on
   * self medium
   */
  EdgeNormalView get_edge_normal() const { return medium1_edge_normal; }
};
} // namespace compute
} // namespace specfem
