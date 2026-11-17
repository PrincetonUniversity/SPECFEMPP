#pragma once

#include "specfem/data_access/accessor.hpp"
#include "specfem/datatype/point_view.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/element_connections/tags.hpp"
#include "specfem/element_coupling/tags.hpp"

namespace specfem::point {

/**
 * @brief Coupling data of one host node of a nonconforming interface pair.
 *
 * Holds the integration factor and normal at the host node, and the 1D
 * Lagrange polynomials of the guest face evaluated at the node's location on
 * that face. The guest shape function \f$ (k_i, k_j) \f$ takes the value
 * `interpolants(k_i, 0) * interpolants(k_j, 1)` at the node.
 *
 * @tparam DimensionTag Spatial dimension
 * @tparam NGLL Number of GLL points per axis of the guest face
 * @tparam InterfaceTag Interface medium type tag
 * @tparam BoundaryTag Boundary condition tag
 * @tparam FluxSchemeTag Flux scheme tag
 */
template <specfem::element::dimension_tag DimensionTag, int NGLL,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag>
struct nonconforming_interface;

template <int NGLL, specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag>
struct nonconforming_interface<specfem::element::dimension_tag::dim3, NGLL,
                               InterfaceTag, BoundaryTag, FluxSchemeTag>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::element::dimension_tag::dim3, false /*UseSIMD*/> {

public:
  /// Spatial dimension (3D for this specialization)
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;
  /// Interface medium type tag
  static constexpr auto interface_tag = InterfaceTag;
  /// Boundary condition tag
  static constexpr auto boundary_tag = BoundaryTag;
  /// Flux scheme tag
  static constexpr auto flux_scheme_tag = FluxSchemeTag;
  /// Connection type for nonconforming interfaces
  static constexpr auto connection_tag =
      specfem::element_connections::type::nonconforming;

  /// Spatial dimension
  static constexpr int ndim = specfem::element::dimension<dimension_tag>::dim;
  /// Number of GLL points per axis of the guest face
  static constexpr int ngll = NGLL;

  /// Integration factor at the node
  scalar_type<type_real> face_factor;
  /// Face normal at the node
  vector_type<type_real, ndim> face_normal;
  /// Guest-face 1D Lagrange polynomials at the node: (igll, axis)
  tensor_type<type_real, NGLL, ndim - 1> interpolants;

  /**
   * @brief Constructs the point from its stored values.
   *
   * @param face_factor Integration factor at the node
   * @param face_normal Face normal at the node
   * @param interpolants Guest-face 1D Lagrange polynomials at the node
   */
  KOKKOS_INLINE_FUNCTION nonconforming_interface(
      const scalar_type<type_real> &face_factor,
      const vector_type<type_real, ndim> &face_normal,
      const tensor_type<type_real, NGLL, ndim - 1> &interpolants)
      : face_factor(face_factor), face_normal(face_normal),
        interpolants(interpolants) {}

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  nonconforming_interface() = default;

  /**
   * @brief Get shared memory size requirement
   * @return Size in bytes needed for scratch memory
   */
  constexpr static int shmem_size() { return 0; }
};

} // namespace specfem::point
