#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::point {

/**
 * @struct conforming_interface
 * @brief Primary template for coupled interface points
 *
 * @tparam DimensionTag Spatial dimension
 * @tparam InterfaceTag Interface type (elastic-acoustic, acoustic-elastic)
 * @tparam BoundaryTag Boundary condition type
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct conforming_interface;

/**
 * @brief 2D coupled interface point data structure
 *
 * Represents a point on a coupled interface between different physical
 * media in 2D spectral element simulations. Contains geometric data
 * (edge factor and normal vector) needed for interface computations.
 *
 * @tparam InterfaceTag Type of interface (elastic-acoustic or acoustic-elastic)
 * @tparam BoundaryTag Boundary condition applied to the interface
 */
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct conforming_interface<specfem::element::dimension_tag::dim2, InterfaceTag,
                            BoundaryTag>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::conforming_interface,
          specfem::element::dimension_tag::dim2, false> {
private:
  /**
   * @brief Base accessor type alias.
   */
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::conforming_interface,
      specfem::element::dimension_tag::dim2, false>;

public:
  /**
   * @brief Dimension tag for 2D specialization.
   */
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;

  /**
   * @brief Connection type between elements.
   */
  static constexpr auto connection_tag =
      specfem::connections::type::weakly_conforming;

  /**
   * @brief Interface type (elastic-acoustic or acoustic-elastic).
   */
  static constexpr auto interface_tag = InterfaceTag;

  /**
   * @brief Boundary condition type.
   */
  static constexpr auto boundary_tag = BoundaryTag;

  /**
   * @brief Edge scaling factor for interface computations.
   */
  scalar_type<type_real> edge_factor;

  /**
   * @brief Edge normal vector (2D).
   */
  vector_type<type_real, 2> edge_normal;

  /**
   * @brief Constructs coupled interface point with geometric data.
   *
   * @param edge_factor Scaling factor for the interface edge.
   * @param edge_normal_ Normal vector at the interface edge.
   */
  KOKKOS_INLINE_FUNCTION
  conforming_interface(const scalar_type<type_real> &edge_factor,
                       const vector_type<type_real, 2> &edge_normal_)
      : edge_factor(edge_factor), edge_normal(edge_normal_) {}

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  conforming_interface() = default;
};

} // namespace specfem::point
