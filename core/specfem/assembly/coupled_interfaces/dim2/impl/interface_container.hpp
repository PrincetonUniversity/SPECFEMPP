
#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"

namespace specfem::assembly::coupled_interfaces_impl {

/**
 * @brief Container for 2D coupled interface data storage and access
 *
 * Manages interface data between different physical media (elastic-acoustic)
 * with specific boundary conditions. Stores edge factors and normal vectors
 * for interface computations in 2D spectral element simulations.
 *
 * @tparam InterfaceTag Type of interface (ELASTIC_ACOUSTIC or ACOUSTIC_ELASTIC)
 * @tparam BoundaryTag Boundary condition type (NONE, STACEY, etc.)
 */
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct interface_container<specfem::dimension::type::dim2, InterfaceTag,
                           BoundaryTag>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::edge,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2> {
public:
  /** @brief Dimension tag for 2D specialization */
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;
  /** @brief Interface type (elastic-acoustic or acoustic-elastic) */
  constexpr static auto interface_tag = InterfaceTag;
  /** @brief Boundary condition type */
  constexpr static auto boundary_tag = BoundaryTag;
  /** @brief Medium type on the self side of the interface */
  constexpr static auto self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();
  /** @brief Medium type on the coupled side of the interface */
  constexpr static auto coupled_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::coupled_medium();

private:
  /** @brief Base container type alias */
  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::edge,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2>;
  /** @brief View type for edge scaling factors */
  using EdgeFactorView = typename base_type::scalar_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  /** @brief View type for edge normal vectors */
  using EdgeNormalView = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  /** @brief Device view for edge scaling factors */
  EdgeFactorView edge_factor;
  /** @brief Device view for edge normal vectors */
  EdgeNormalView edge_normal;

  /** @brief Host mirror for edge scaling factors */
  EdgeFactorView::HostMirror h_edge_factor;
  /** @brief Host mirror for edge normal vectors */
  EdgeNormalView::HostMirror h_edge_normal;

public:
  /**
   * @brief Constructs interface container with mesh and geometry data
   *
   * @param ngllz Number of GLL points in z-direction
   * @param ngllx Number of GLL points in x-direction
   * @param edge_types Edge type information from mesh
   * @param jacobian_matrix Jacobian transformation data
   * @param mesh Mesh connectivity and geometry
   */
  interface_container(
      const int ngllz, const int ngllx,
      const specfem::assembly::edge_types<specfem::dimension::type::dim2>
          &edge_types,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::mesh<dimension_tag> &mesh);

  /** @brief Default constructor */
  interface_container() = default;

  /**
   * @brief Loads interface data at specified index into point
   *
   * Template function that loads edge factor and normal vector data
   * from either device or host memory into the provided point object.
   *
   * @tparam on_device If true, loads from device memory; if false, from host
   * @tparam IndexType Type of index (must have iedge and ipoint members)
   * @tparam PointType Type of point (must have edge_factor and edge_normal)
   * @param index Edge and point indices for data location
   * @param point Output point object to store loaded data
   */
  template <bool on_device, typename IndexType, typename PointType>
  KOKKOS_FORCEINLINE_FUNCTION void impl_load(const IndexType &index,
                                             PointType &point) const {
    if constexpr (on_device) {
      point.edge_factor = edge_factor(index.iedge, index.ipoint);
      point.edge_normal(0) = edge_normal(index.iedge, index.ipoint, 0);
      point.edge_normal(1) = edge_normal(index.iedge, index.ipoint, 1);
    } else {
      point.edge_factor = h_edge_factor(index.iedge, index.ipoint);
      point.edge_normal(0) = h_edge_normal(index.iedge, index.ipoint, 0);
      point.edge_normal(1) = h_edge_normal(index.iedge, index.ipoint, 1);
    }
    return;
  }
};
} // namespace specfem::assembly::coupled_interfaces_impl
