#ifndef _ENUMS_BOUNDARY_CONDITIONS_STACEY2D_ELASTIC_TPP_
#define _ENUMS_BOUNDARY_CONDITIONS_STACEY2D_ELASTIC_TPP_

#include "algorithms/dot.hpp"
#include "compute/interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "stacey2d_elastic.hpp"
#include <Kokkos_Core.hpp>

namespace {
KOKKOS_FUNCTION void enforce_traction_boundary(
    const type_real &weight,
    const specfem::datatype::ScalarPointViewType<type_real, 2> &dn,
    const specfem::point::properties<specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
                                     specfem::element::property_tag::isotropic>
        &properties,
    const specfem::datatype::ScalarPointViewType<type_real, 2> &field_dot,
    specfem::datatype::ScalarPointViewType<type_real, 2> &field_dot_dot) {

  auto jacobian1d = dn.l2_norm();

  auto vn = specfem::algorithms::dot(dn, field_dot);

  type_real traction[2];

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int icomp = 0; icomp < 2; ++icomp)
    traction[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                       (properties.rho_vp - properties.rho_vs)) +
                      field_dot(icomp) * properties.rho_vs;

  field_dot_dot(0) += -1.0 * traction[0] * jacobian1d * weight;
  field_dot_dot(1) += -1.0 * traction[1] * jacobian1d * weight;

  return;
}

template <specfem::element::property_tag property>
KOKKOS_FUNCTION void newmark_mass_terms(
    const int &ix, const int &iz, const int &ngllx, const int &ngllz,
    const type_real &dt, const specfem::point::boundary &boundary_type,
    const specfem::element::boundary_tag &tag,
    const specfem::datatype::ScalarPointViewType<type_real, 2> &weight,
    const specfem::point::partial_derivatives2<true> &partial_derivatives,
    const specfem::point::properties<specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
                                     property> &properties,
    specfem::datatype::ScalarPointViewType<type_real, 2> &rmass_inverse) {

  specfem::datatype::ScalarPointViewType<type_real, 2> velocity(
      static_cast<type_real>(-1.0 * dt * 0.5));

  const specfem::enums::edge::type edge = [&]() -> specfem::enums::edge::type {
    if (boundary_type.left == tag && ix == 0)
      return specfem::enums::edge::type::LEFT;
    if (boundary_type.right == tag && ix == ngllx - 1)
      return specfem::enums::edge::type::RIGHT;
    if (boundary_type.top == tag && iz == ngllz - 1)
      return specfem::enums::edge::type::TOP;
    if (boundary_type.bottom == tag && iz == 0)
      return specfem::enums::edge::type::BOTTOM;
    return specfem::enums::edge::type::NONE;
  }();

  const type_real factor = [&]() -> type_real {
    switch (edge) {
    case specfem::enums::edge::type::LEFT:
    case specfem::enums::edge::type::RIGHT:
      return weight(1);
      break;
    case specfem::enums::edge::type::TOP:
    case specfem::enums::edge::type::BOTTOM:
      return weight(0);
      break;
    default:
      return static_cast<type_real>(0.0);
      break;
    }
  }();

  if (edge == specfem::enums::edge::type::NONE)
    return;

  const auto dn = partial_derivatives.compute_normal(edge);
  enforce_traction_boundary(factor, dn, properties, velocity, rmass_inverse);

  // specfem::kokkos::array_type<type_real, 2> dn;

  // // Left Boundary
  // if (boundary_type.left == tag && ix == 0) {
  //   dn = partial_derivatives
  //            .compute_normal<specfem::enums::boundaries::type::LEFT>();
  //   enforce_traction_boundary(weight[1], dn, properties, velocity,
  //                             rmass_inverse);
  //   return;
  // }

  // // Right Boundary
  // if (boundary_type.right == tag && ix == ngllx - 1) {
  //   dn = partial_derivatives
  //            .compute_normal<specfem::enums::boundaries::type::RIGHT>();
  //   enforce_traction_boundary(weight[1], dn, properties, velocity,
  //                             rmass_inverse);
  //   return;
  // }

  // // Top Boundary
  // if (boundary_type.top == tag && iz == ngllz - 1) {
  //   dn = partial_derivatives
  //            .compute_normal<specfem::enums::boundaries::type::TOP>();
  //   enforce_traction_boundary(weight[0], dn, properties, velocity,
  //                             rmass_inverse);
  //   return;
  // }

  // // Bottom Boundary
  // if (boundary_type.bottom == tag && iz == 0) {
  //   dn = partial_derivatives
  //            .compute_normal<specfem::enums::boundaries::type::BOTTOM>();
  //   enforce_traction_boundary(weight[0], dn, properties, velocity,
  //                             rmass_inverse);
  //   return;
  // }

  return;
}
} // namespace

// template <typename property, typename qp_type>
// specfem::enums::boundary_conditions::stacey<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::elastic, property,
//     qp_type>::stacey(const specfem::compute::boundaries &boundary_conditions,
//                      const quadrature_points_type &quadrature_points)
//     : quadrature_points(quadrature_points),
//       type(boundary_conditions.stacey.elastic.type) {
//   return;
// }

template <specfem::wavefield::type WavefieldType,
          specfem::element::property_tag PropertyTag, typename qp_type>
template <specfem::enums::time_scheme::type time_scheme>
KOKKOS_INLINE_FUNCTION void
specfem::boundary::boundary<WavefieldType, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic, PropertyTag,
                            specfem::element::boundary_tag::stacey, qp_type>::
    mass_time_contribution(
        const int &xz, const type_real &dt,
        const specfem::datatype::ScalarPointViewType<type_real, 2> &weight,
        const specfem::point::partial_derivatives2<true> &partial_derivatives,
        const specfem::point::properties<specfem::dimension::type::dim2, medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::boundary &boundary_type,
        specfem::datatype::ScalarPointViewType<type_real, medium_type::components>
            &rmass_inverse) const {

  // Check if the GLL point is on the boundary
  //--------------------------------------------------------------------------
  constexpr int components = 2;
  constexpr auto value_t = value;

  int ngllx, ngllz;

  quadrature_points.get_ngll(&ngllx, &ngllz);

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);

  if (!specfem::point::is_on_boundary(value_t, boundary_type, iz, ix, ngllz,
                                      ngllx)) {
    return;
  }
  //--------------------------------------------------------------------------

  if constexpr (time_scheme == specfem::enums::time_scheme::type::newmark) {
    newmark_mass_terms(ix, iz, ngllx, ngllz, dt, boundary_type, value_t, weight,
                       partial_derivatives, properties, rmass_inverse);
    return;
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::property_tag PropertyTag, typename qp_type>
KOKKOS_INLINE_FUNCTION void
specfem::boundary::boundary<WavefieldType, specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic, PropertyTag,
                            specfem::element::boundary_tag::stacey, qp_type>::
    enforce_traction(
        const int &xz, const specfem::datatype::ScalarPointViewType<type_real, 2> &weight,
        const specfem::point::partial_derivatives2<true> &partial_derivatives,
        const specfem::point::properties<specfem::dimension::type::dim2, medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::boundary &boundary_type,
        const specfem::datatype::ScalarPointViewType<type_real, medium_type::components>
            &field_dot,
        specfem::datatype::ScalarPointViewType<type_real, medium_type::components>
            &field_dot_dot) const {

  // Check if the GLL point is on the boundary
  //--------------------------------------------------------------------------
  constexpr int components = 2;
  constexpr auto value_t = value;

  int ngllx, ngllz;

  quadrature_points.get_ngll(&ngllx, &ngllz);

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);

  if (!specfem::point::is_on_boundary(value_t, boundary_type, iz, ix, ngllz,
                                      ngllx)) {
    return;
  }
  //--------------------------------------------------------------------------

  // enforce traction condition
  // --------------------------------------------------------------------------
  // If the GLL point is on the corner the left or right traction conditions are
  // applied top or bottom traction conditions are ignored in this case. This
  // ensures there is no conflict in calculating the normal

  const specfem::enums::edge::type edge = [&]() -> specfem::enums::edge::type {
    if (boundary_type.left == value_t && ix == 0)
      return specfem::enums::edge::type::LEFT;
    if (boundary_type.right == value_t && ix == ngllx - 1)
      return specfem::enums::edge::type::RIGHT;
    if (boundary_type.top == value_t && iz == ngllz - 1)
      return specfem::enums::edge::type::TOP;
    if (boundary_type.bottom == value_t && iz == 0)
      return specfem::enums::edge::type::BOTTOM;
    return specfem::enums::edge::type::NONE;
  }();

  const type_real factor = [&]() -> type_real {
    switch (edge) {
    case specfem::enums::edge::type::LEFT:
    case specfem::enums::edge::type::RIGHT:
      return weight(1);
      break;
    case specfem::enums::edge::type::TOP:
    case specfem::enums::edge::type::BOTTOM:
      return weight(0);
      break;
    default:
      return static_cast<type_real>(0.0);
      break;
    }
  }();

  if (edge == specfem::enums::edge::type::NONE)
    return;

  const auto dn = partial_derivatives.compute_normal(edge);
  enforce_traction_boundary(factor, dn, properties, field_dot, field_dot_dot);

  // specfem::kokkos::array_type<type_real, dimension::dim> dn;

  // // Left Boundary
  // if (itype.left == value_t && ix == 0) {
  //   dn = partial_derivatives
  //            .compute_normal<specfem::enums::boundaries::type::LEFT>();
  //   enforce_traction_boundary(weight[1], dn, properties, field_dot,
  //                             field_dot_dot);
  //   return;
  // }

  // // Right Boundary
  // if (itype.right == value_t && ix == ngllx - 1) {
  //   dn = partial_derivatives
  //            .compute_normal<specfem::enums::boundaries::type::RIGHT>();
  //   enforce_traction_boundary(weight[1], dn, properties, field_dot,
  //                             field_dot_dot);
  //   return;
  // }

  // // Top Boundary
  // if (itype.top == value_t && iz == ngllz - 1) {
  //   dn = partial_derivatives
  //            .compute_normal<specfem::enums::boundaries::type::TOP>();
  //   enforce_traction_boundary(weight[0], dn, properties, field_dot,
  //                             field_dot_dot);
  //   return;
  // }

  // // Bottom Boundary
  // if (itype.bottom == value_t && iz == 0) {
  //   dn = partial_derivatives
  //            .compute_normal<specfem::enums::boundaries::type::BOTTOM>();
  //   enforce_traction_boundary(weight[0], dn, properties, field_dot,
  //                             field_dot_dot);
  //   return;
  // }
  // --------------------------------------------------------------------------

  return;
}

#endif /* _ENUMS_BOUNDARY_CONDITIONS_STACEY2D_ELASTIC_TPP_ */
