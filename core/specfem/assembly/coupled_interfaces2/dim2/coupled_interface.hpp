#pragma once

#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "impl/interface_container.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"

namespace specfem::assembly {

/**
 * @brief 2D coupled interfaces container for spectral element computations.
 *
 * This class manages the storage and access of data required to compute
 * coupling between elements connected via weakly conforming edges in a 2D
 * spectral element mesh. The interface data is split into multiple containers
 * based on the types of media on either side of the interface (e.g.,
 * elastic-acoustic, acoustic-elastic) and the boundary conditions applied
 * (e.g., free surface, Stacey absorbing).
 *
 * @tparam specfem::dimension::type::dim2 Template specialization for 2D domain
 *
 * @note This is a template specialization for 2D domains. The primary template
 *       is declared elsewhere and specialized here for dimension-specific
 *       optimizations.
 *
 * @see specfem::assembly::coupled_interfaces2_impl::interface_container
 * @see specfem::assembly::edge_types
 * @see specfem::assembly::jacobian_matrix
 * @see specfem::assembly::mesh
 */
template <>
class coupled_interfaces2<specfem::dimension::type::dim2>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::edge,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2> {
public:
  /**
   * @brief Dimension tag for this specialization
   *
   * Static constant member that identifies this specialization as operating
   * in 2D space. Used for compile-time dispatch and type checking.
   */
  static constexpr auto dimension_tag = specfem::dimension::type::dim2;

private:
  template <specfem::interface::interface_tag InterfaceTag,
            specfem::element::boundary_tag BoundaryTag>
  using InterfaceContainerType =
      specfem::assembly::coupled_interfaces2_impl::interface_container<
          dimension_tag, InterfaceTag, BoundaryTag>;

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
                       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
                       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                                    COMPOSITE_STACEY_DIRICHLET)),
                      DECLARE(((InterfaceContainerType,
                                (_INTERFACE_TAG_, _BOUNDARY_TAG_)),
                               interface_container)))

public:
  /**
   * @brief Constructor for 2D coupled interfaces container
   *
   * Initializes all interface containers for the supported combinations of
   * media types and boundary conditions.
   *
   * @param ngllz Number of Gauss-Lobatto-Legendre points in the z-direction
   * @param ngllx Number of Gauss-Lobatto-Legendre points in the x-direction
   * @param edge_types Reference to the edge types container that provides
   *                   information about the types of edges in the mesh
   *                   (e.g., boundary edges, internal edges).
   * @param jacobian_matrix Reference to the Jacobian matrix container that
   *                        provides geometric transformation information
   *                        between reference and physical coordinates.
   * @param mesh Reference to the 2D mesh container that provides element
   *             connectivity, material properties, and geometric information.
   *
   * @pre edge_types must be properly initialized for the given mesh
   * @pre jacobian_matrix must be computed for all elements in the mesh
   * @pre mesh must contain valid element-to-node connectivity
   *
   * @post All interface containers are initialized and ready for use
   * @post Memory is allocated for all supported interface combinations
   * @see specfem::assembly::edge_types
   * @see specfem::assembly::jacobian_matrix
   * @see specfem::assembly::mesh
   */
  coupled_interfaces2(
      const int ngllz, const int ngllx,
      const specfem::assembly::edge_types<dimension_tag> &edge_types,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::mesh<dimension_tag> &mesh);

  /**
   * @brief Friend function for loading interface data on host
   *
   * Provides access to the private interface containers for the host-side
   * data loading operations. This friend declaration allows the external
   * load_on_host function to access the internal container members.
   *
   * @tparam IndexType Type of the index used to specify the interface location
   * @tparam ContainerType Type of the container holding the interface data
   * @tparam PointType Type representing the interface point data structure
   */
  template <typename IndexType, typename ContainerType, typename PointType>
  friend void load_on_host(const IndexType &index, ContainerType &container,
                           const PointType &point);

  /**
   * @brief Friend function for loading interface data on device
   *
   * Provides access to the private interface containers for the device-side
   * data loading operations. This friend declaration allows the external
   * load_on_device function to access the internal container members.
   * The function is marked with KOKKOS_FORCEINLINE_FUNCTION for optimal
   * performance on GPU devices.
   *
   * @tparam IndexType Type of the index used to specify the interface location
   * @tparam ContainerType Type of the container holding the interface data
   * @tparam PointType Type representing the interface point data structure
   */
  template <typename IndexType, typename ContainerType, typename PointType>
  friend KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const IndexType &index, ContainerType &container,
                 const PointType &point);
};

/**
 * @defgroup CoupledInterfaceDataAccess
 * @brief Data access functions for coupled interface computation data
 *
 */

/**
 * @brief Load interface data from container to point on host
 *
 * Loads coupled interface data using compile-time dispatch based on the point's
 * template parameters (connection, interface, and boundary types).
 *
 * @ingroup CoupledInterfaceDataAccess
 *
 * @tparam IndexType Edge index type
 * @tparam ContainerType Coupled interfaces container type
 * @tparam PointType Interface point type
 *
 * @param index Edge index specifying the interface location
 * @param container Coupled interfaces container holding interface data
 * @param point Point object where loaded data will be stored
 *
 * @pre index refers to valid mesh edge
 * @pre container is properly initialized
 * @pre point type matches supported interface combinations
 *
 * @note For host-side computations only. Use load_on_device for device code.
 */
template <
    typename IndexType, typename ContainerType, typename PointType,
    typename std::enable_if_t<
        ((specfem::data_access::is_edge_index<IndexType>::value) &&
         (specfem::data_access::is_coupled_interface<ContainerType>::value)),
        int> = 0>
inline void load_on_host(const IndexType &index, ContainerType &container,
                         const PointType &point) {

  static_assert(
      specfem::data_access::CheckCompatibility<IndexType, ContainerType,
                                               PointType>::value,
      "Incompatible types in load_on_host");

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE((interface_container, container.interface_container)) {
        if constexpr (PointType::connection_tag == _connection_tag_ &&
                      PointType::interface_tag == _interface_tag_ &&
                      PointType::boundary_tag == _boundary_tag_) {
          _interface_container_.impl_load<false>(index, point);
          return;
        }
      })

#ifndef NDEBUG
  KOKKOS_ABORT_WITH_LOCATION(
      "specfem::assembly::load_on_host(): No matching specialization found.");
#endif
}

/**
 * @brief Load interface data from container to point on device
 *
 * Loads coupled interface data using compile-time dispatch based on the point's
 * template parameters (connection, interface, and boundary types).
 *
 * @ingroup CoupledInterfaceDataAccess
 *
 * @tparam IndexType Edge index type
 * @tparam ContainerType Coupled interfaces container type
 * @tparam PointType Interface point type
 *
 * @param index Edge index specifying the interface location
 * @param container Coupled interfaces container holding interface data
 * @param point Point object where loaded data will be stored
 *
 * @pre index refers to valid mesh edge
 * @pre container is properly initialized
 * @pre point type matches supported interface combinations
 *
 * @note For device-side computations only. Use load_on_host for host code.
 */
template <
    typename IndexType, typename ContainerType, typename PointType,
    typename std::enable_if_t<
        ((specfem::data_access::is_edge_index<IndexType>::value) &&
         (specfem::data_access::is_coupled_interface<ContainerType>::value)),
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                ContainerType &container,
                                                const PointType &point) {

  static_assert(
      specfem::data_access::CheckCompatibility<IndexType, ContainerType,
                                               PointType>::value,
      "Incompatible types in load_on_host");

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE((interface_container, container.interface_container)) {
        if constexpr (PointType::connection_tag == _connection_tag_ &&
                      PointType::interface_tag == _interface_tag_ &&
                      PointType::boundary_tag == _boundary_tag_) {
          _interface_container_.impl_load<true>(index, point);
          return;
        }
      })

#ifndef NDEBUG
  KOKKOS_ABORT_WITH_LOCATION("specfem::assembly::load_on_device(): No "
                             "matching specialization found.");
#endif
}

} // namespace specfem::assembly
