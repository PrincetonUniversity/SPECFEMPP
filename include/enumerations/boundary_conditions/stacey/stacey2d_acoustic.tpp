#ifndef _ENUMS_BOUNDARY_CONDITIONS_STACEY_2D_ACOUSTIC_TPP_
#define _ENUMS_BOUNDARY_CONDITIONS_STACEY_2D_ACOUSTIC_TPP_

#include "compute/interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "stacey2d_acoustic.hpp"

namespace {

KOKKOS_INLINE_FUNCTION void enforce_traction_boundary(
    const type_real &weight,
    const specfem::kokkos::array_type<type_real, 2> &dn,
    const specfem::point::properties<specfem::element::medium_tag::acoustic,
                                     specfem::element::property_tag::isotropic>
        &properties,
    const specfem::kokkos::array_type<type_real, 1> &field_dot,
    specfem::kokkos::array_type<type_real, 1> &field_dot_dot) {

  auto jacobian1d = dn.l2_norm();

  field_dot_dot[0] +=
      // jacobian1d * weight * rho_vp_inverse * velocity
      -1.0 * dn.l2_norm() * weight * properties.rho_vpinverse * field_dot[0];
  return;
}

template <specfem::element::property_tag property>
KOKKOS_FUNCTION void newmark_mass_terms(
    const int &ix, const int &iz, const int &ngllx, const int &ngllz,
    const type_real &dt, const specfem::point::boundary &boundary_type,
    const specfem::element::boundary_tag &tag,
    const specfem::kokkos::array_type<type_real, 2> &weight,
    const specfem::point::partial_derivatives2 &partial_derivatives,
    const specfem::point::properties<specfem::element::medium_tag::acoustic,
                                     property> &properties,
    specfem::kokkos::array_type<type_real, 1> &rmass_inverse) {

  const specfem::kokkos::array_type<type_real, 1> velocity(
      static_cast<type_real>(-1.0 * dt * 0.5));

  const specfem::enums::edge::type edge = [&]() -> specfem::enums::edge::type {
    if (boundary_type.left == tag && ix == 0)
      return specfem::enums::edge::type::LEFT;
    if (boundary_type.right == tag && ix == ngllx - 1)
      return specfem::enums::edge::type::RIGHT;
    if (boundary_type.bottom == tag && iz == 0)
      return specfem::enums::edge::type::BOTTOM;
    if (boundary_type.top == tag && iz == ngllz - 1)
      return specfem::enums::edge::type::TOP;
    return specfem::enums::edge::type::NONE;
  }();

  const type_real factor = [&]() -> type_real {
    switch (edge) {
    case specfem::enums::edge::type::LEFT:
    case specfem::enums::edge::type::RIGHT:
      return weight[1];
      break;
    case specfem::enums::edge::type::BOTTOM:
    case specfem::enums::edge::type::TOP:
      return weight[0];
      break;
    default:
      return static_cast<type_real>(0.0);
      break;
    }
  }();

  if (edge == specfem::enums::edge::type::NONE) {
    return;
  }

  const auto dn = partial_derivatives.compute_normal(edge);
  enforce_traction_boundary(factor, dn, properties, velocity, rmass_inverse);

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

  // // Bottom Boundary
  // if (boundary_type.bottom == tag && iz == 0) {
  //   dn = partial_derivatives
  //            .compute_normal<specfem::enums::boundaries::type::BOTTOM>();
  //   enforce_traction_boundary(weight[0], dn, properties, velocity,
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

  return;
}
} // namespace

// template <typename property, typename qp_type>
// specfem::enums::boundary_conditions::stacey<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::acoustic, property,
//     qp_type>::stacey(const specfem::compute::boundaries &boundary_conditions,
//                      const quadrature_points_type &quadrature_points)
//     : quadrature_points(quadrature_points),
//       type(boundary_conditions.stacey.acoustic.type) {
//   return;
// }

template <specfem::element::property_tag property, typename qp_type>
template <specfem::enums::time_scheme::type time_scheme>
KOKKOS_INLINE_FUNCTION void
specfem::boundary::boundary<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic, property,
                            specfem::element::boundary_tag::stacey, qp_type>::
    mass_time_contribution(
        const int &xz, const type_real &dt,
        const specfem::kokkos::array_type<type_real, 2> &weight,
        const specfem::point::partial_derivatives2 &partial_derivatives,
        const specfem::point::properties<medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::boundary &boundary_type,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &rmass_inverse) const {

  // Check if the GLL point is on the boundary
  // --------------------------------------------------------------------------

  constexpr int components = 1;
  constexpr auto value_t = value;

  int ngllx, ngllz;

  quadrature_points.get_ngll(&ngllx, &ngllz);

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);

  if (!specfem::point::is_on_boundary(value_t, boundary_type, iz, ix, ngllz,
                                      ngllx)) {
    return;
  }
  // --------------------------------------------------------------------------

  if constexpr (time_scheme == specfem::enums::time_scheme::type::newmark) {
    newmark_mass_terms(ix, iz, ngllx, ngllz, dt, boundary_type, value_t, weight,
                       partial_derivatives, properties, rmass_inverse);
    return;
  }

  return;
}

template <specfem::element::property_tag property, typename qp_type>
KOKKOS_INLINE_FUNCTION void
specfem::boundary::boundary<specfem::dimension::type::dim2,
                            specfem::element::medium_tag::acoustic, property,
                            specfem::element::boundary_tag::stacey, qp_type>::
    enforce_traction(
        const int &xz, const specfem::kokkos::array_type<type_real, 2> &weight,
        const specfem::point::partial_derivatives2 &partial_derivatives,
        const specfem::point::properties<medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::boundary &boundary_type,
        const specfem::kokkos::array_type<type_real, medium_type::components>
            &field_dot,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &field_dot_dot) const {

  // Check if the GLL point is on the boundary
  // --------------------------------------------------------------------------
  constexpr int components = 1;
  constexpr auto value_t = value;

  int ngllx, ngllz;

  quadrature_points.get_ngll(&ngllx, &ngllz);

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);

  if (!specfem::point::is_on_boundary(value_t, boundary_type, iz, ix, ngllz,
                                      ngllx)) {
    return;
  }
  // --------------------------------------------------------------------------

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
    if (boundary_type.bottom == value_t && iz == 0)
      return specfem::enums::edge::type::BOTTOM;
    if (boundary_type.top == value_t && iz == ngllz - 1)
      return specfem::enums::edge::type::TOP;
    return specfem::enums::edge::type::NONE;
  }();

  const type_real factor = [&]() -> type_real {
    switch (edge) {
    case specfem::enums::edge::type::LEFT:
    case specfem::enums::edge::type::RIGHT:
      return weight[1];
      break;
    case specfem::enums::edge::type::BOTTOM:
    case specfem::enums::edge::type::TOP:
      return weight[0];
      break;
    default:
      return static_cast<type_real>(0.0);
      break;
    }
  }();

  if (edge == specfem::enums::edge::type::NONE) {
    return;
  }

  const auto dn = partial_derivatives.compute_normal(edge);
  enforce_traction_boundary(factor, dn, properties, field_dot, field_dot_dot);

  // specfem::kokkos::array_type<type_real, dimension::dim> dn; // normal vector

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

  // // Bottom Boundary
  // if (itype.bottom == value_t && iz == 0) {
  //   dn = partial_derivatives
  //            .compute_normal<specfem::enums::boundaries::type::BOTTOM>();
  //   enforce_traction_boundary(weight[0], dn, properties, field_dot,
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
  // --------------------------------------------------------------------------

  return;
}

#endif /* _ENUMS_BOUNDARY_CONDITIONS_STACEY_2D_ACOUSTIC_TPP_ */
