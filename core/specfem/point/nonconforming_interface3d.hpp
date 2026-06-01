#pragma once

#include "specfem/data_access/accessor.hpp"
#include "specfem/datatype/point_view.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/element_connections/tags.hpp"
#include "specfem/element_coupling/tags.hpp"

namespace specfem::point {

/**
 * @brief Template accessor for face node location in coupled local coordinates.
 *
 * @tparam DimensionTag
 * @tparam NGLL number of quadrature points, used only for the interpolants
 * @tparam InterfaceTag
 * @tparam BoundaryTag
 * @tparam FluxSchemeTag
 * @tparam MemorySpace
 * @tparam MemoryTraits
 */
template <specfem::element::dimension_tag DimensionTag,
          int NGLL /*Only for interpolants*/,
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
  /// Spatial dimension (2D for this specialization)
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

  static constexpr int ndim = specfem::element::dimension<dimension_tag>::dim;
  vector_type<type_real, ndim - 1> coupled_coordinates;
  scalar_type<type_real> face_factor;
  vector_type<type_real, ndim> face_normal;

  // ========================= START TEMPORARY =========================
  tensor_type<type_real, NGLL, ndim - 1> interpolants;
  template <typename LagrangeInterpolantType>
  KOKKOS_INLINE_FUNCTION nonconforming_interface(
      const vector_type<type_real, ndim - 1> &coupled_coordinates,
      const scalar_type<type_real> &face_factor,
      const vector_type<type_real, ndim> &face_normal,
      const LagrangeInterpolantType &lagrange_interpolant)
      : coupled_coordinates(coupled_coordinates), face_factor(face_factor),
        face_normal(face_normal) {
    // populate interpolants array.
    for (int igll = 0; igll < NGLL; igll++) {
      for (int idim = 0; idim < ndim; idim++) {
        interpolants(igll, idim) =
            lagrange_interpolant(igll, coupled_coordinates(idim));
      }
    }
  }

  // =========================  END TEMPORARY  =========================

  KOKKOS_INLINE_FUNCTION nonconforming_interface(
      const vector_type<type_real, ndim - 1> &coupled_coordinates,
      const scalar_type<type_real> &face_factor,
      const vector_type<type_real, ndim> &face_normal)
      : coupled_coordinates(coupled_coordinates), face_factor(face_factor),
        face_normal(face_normal) {}

  /**
   * @brief Default constructor.
   */
  KOKKOS_INLINE_FUNCTION
  nonconforming_interface() = default;
};

} // namespace specfem::point
