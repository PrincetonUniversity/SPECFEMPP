#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::point {

/**
 * @brief Primary template for coupled interface points
 *
 * @tparam DimensionTag Spatial dimension
 * @tparam ConnectionTag Connection type (strongly/weakly conforming)
 * @tparam InterfaceTag Interface type (elastic-acoustic, acoustic-elastic)
 * @tparam BoundaryTag Boundary condition type
 */
template <specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct coupled_interface;

/**
 * @brief 2D coupled interface point data structure
 *
 * Represents a point on a coupled interface between different physical
 * media in 2D spectral element simulations. Contains geometric data
 * (edge factor and normal vector) needed for interface computations.
 *
 * @tparam ConnectionTag Connection type between elements
 * @tparam InterfaceTag Type of interface (elastic-acoustic or acoustic-elastic)
 * @tparam BoundaryTag Boundary condition applied to the interface
 */
template <specfem::connections::type ConnectionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct coupled_interface<specfem::dimension::type::dim2, ConnectionTag,
                         InterfaceTag, BoundaryTag>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2, false> {
private:
  /** @brief Base accessor type alias */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2, false>;

public:
  /** @brief Dimension tag for 2D specialization */
  static constexpr auto dimension_tag = specfem::dimension::type::dim2;
  /** @brief Connection type between elements */
  static constexpr auto connection_tag = ConnectionTag;
  /** @brief Interface type (elastic-acoustic or acoustic-elastic) */
  static constexpr auto interface_tag = InterfaceTag;
  /** @brief Boundary condition type */
  static constexpr auto boundary_tag = BoundaryTag;

  /** @brief Edge scaling factor for interface computations */
  scalar_type<type_real> edge_factor;
  /** @brief Edge normal vector (2D) */
  vector_type<type_real, 2> edge_normal;

  /**
   * @brief Constructs coupled interface point with geometric data
   *
   * @param edge_factor Scaling factor for the interface edge
   * @param edge_normal_ Normal vector at the interface edge
   */
  KOKKOS_INLINE_FUNCTION
  coupled_interface(const scalar_type<type_real> &edge_factor,
                    const vector_type<type_real, 2> &edge_normal_)
      : edge_factor(edge_factor), edge_normal(edge_normal_) {}

  KOKKOS_INLINE_FUNCTION
  coupled_interface() = default;
};

} // namespace specfem::point
