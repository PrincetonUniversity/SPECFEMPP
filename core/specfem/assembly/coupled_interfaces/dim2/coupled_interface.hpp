#pragma once

#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "impl/interface_container.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

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
 * @see specfem::assembly::coupled_interfaces_impl::interface_container
 * @see specfem::assembly::edge_types
 * @see specfem::assembly::jacobian_matrix
 * @see specfem::assembly::mesh
 */
template <>
class coupled_interfaces<specfem::dimension::type::dim2>
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
      specfem::assembly::coupled_interfaces_impl::interface_container<
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
  coupled_interfaces(
      const int ngllz, const int ngllx,
      const specfem::assembly::edge_types<dimension_tag> &edge_types,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::mesh<dimension_tag> &mesh);

  coupled_interfaces() = default;

  /**
   * @brief Get interface container for specific coupling and boundary types
   *
   * Uses compile-time dispatch to return the appropriate interface container
   * without runtime overhead. Supports elastic_acoustic/acoustic_elastic
   * interfaces with
   * none/acoustic_free_surface/stacey/composite_stacey_dirichlet boundary
   * conditions.
   *
   * @tparam InterfaceTag Interface coupling type
   * @tparam BoundaryTag Boundary condition type
   * @return const reference to the requested interface container
   *
   * @example
   * ```cpp
   * const auto& container = interfaces.get_interface_container<
   *     specfem::interface::interface_tag::elastic_acoustic,
   *     specfem::element::boundary_tag::stacey>();
   * ```
   */
  template <specfem::interface::interface_tag InterfaceTag,
            specfem::element::boundary_tag BoundaryTag>
  KOKKOS_INLINE_FUNCTION const
      InterfaceContainerType<InterfaceTag, BoundaryTag> &
      get_interface_container() const {
    // Compile-time dispatch using FOR_EACH_IN_PRODUCT macro
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
                         INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
                         BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                                      COMPOSITE_STACEY_DIRICHLET)),
                        CAPTURE((interface_container, interface_container)) {
                          if constexpr (InterfaceTag == _interface_tag_ &&
                                        BoundaryTag == _boundary_tag_) {
                            return _interface_container_;
                          }
                        })

#ifndef NDEBUG
    // Debug check: abort if no matching specialization found
    KOKKOS_ABORT_WITH_LOCATION(
        "specfem::assembly::coupled_interfaces::get_interface_container(): No "
        "matching specialization found.");
#endif

    // Unreachable code - satisfy compiler return requirements
    return {};
  }
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
inline void load_on_host(const IndexType &index, const ContainerType &container,
                         PointType &point) {

  static_assert(
      specfem::data_access::CheckCompatibility<IndexType, ContainerType,
                                               PointType>::value,
      "Incompatible types in load_on_host");

  container
      .template get_interface_container<PointType::interface_tag,
                                        PointType::boundary_tag>()
      .template impl_load<false>(index, point);
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
                                                const ContainerType &container,
                                                PointType &point) {

  static_assert(
      specfem::data_access::CheckCompatibility<IndexType, ContainerType,
                                               PointType>::value,
      "Incompatible types in load_on_host");

  container
      .template get_interface_container<PointType::interface_tag,
                                        PointType::boundary_tag>()
      .template impl_load<true>(index, point);
}

} // namespace specfem::assembly
