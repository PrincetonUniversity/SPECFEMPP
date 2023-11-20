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
    const specfem::compute::element_properties<
        specfem::enums::element::type::acoustic,
        specfem::enums::element::property_tag::isotropic> &properties,
    const specfem::kokkos::array_type<type_real, 1> &field_dot,
    specfem::kokkos::array_type<type_real, 1> &field_dot_dot) {

  auto jacobian1d = dn.l2_norm();

  field_dot_dot[0] +=
      // jacobian1d * weight * rho_vp_inverse * velocity
      -1.0 * dn.l2_norm() * weight * properties.rho_vpinverse * field_dot[0];
  return;
}

template <specfem::enums::element::property_tag property>
KOKKOS_FUNCTION void newmark_mass_terms(
    const int &ix, const int &iz, const int &ngllx, const int &ngllz,
    const type_real &dt, const specfem::compute::access::boundary_types &itype,
    const specfem::kokkos::array_type<type_real, 2> &weight,
    const specfem::compute::element_partial_derivatives &partial_derivatives,
    const specfem::compute::element_properties<
        specfem::enums::element::type::acoustic, property> &properties,
    specfem::kokkos::array_type<type_real, 1> &rmass_inverse) {

  specfem::kokkos::array_type<type_real, 1> velocity;

  velocity[0] = - 1.0 * dt * 0.5;

  specfem::kokkos::array_type<type_real, 2> dn; // normal vector

  // Left Boundary
  if (itype.left && ix == 0) {
    dn = partial_derivatives
             .compute_normal<specfem::enums::boundaries::type::LEFT>();
    enforce_traction_boundary(weight[1], dn, properties, velocity,
                              rmass_inverse);
    return;
  }

  // Right Boundary
  if (itype.right && ix == ngllx - 1) {
    dn = partial_derivatives
             .compute_normal<specfem::enums::boundaries::type::RIGHT>();
    enforce_traction_boundary(weight[1], dn, properties, velocity,
                              rmass_inverse);
    return;
  }

  // Bottom Boundary
  if (itype.bottom && iz == 0) {
    dn = partial_derivatives
             .compute_normal<specfem::enums::boundaries::type::BOTTOM>();
    enforce_traction_boundary(weight[0], dn, properties, velocity,
                              rmass_inverse);
    return;
  }

  // Top Boundary
  if (itype.top && iz == ngllz - 1) {
    dn = partial_derivatives
             .compute_normal<specfem::enums::boundaries::type::TOP>();
    enforce_traction_boundary(weight[0], dn, properties, velocity,
                              rmass_inverse);
    return;
  }

  return;
}
} // namespace

template <typename property, typename qp_type>
specfem::enums::boundary_conditions::stacey<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic, property,
    qp_type>::stacey(const specfem::compute::boundaries &boundary_conditions,
                     const quadrature_points_type &quadrature_points)
    : quadrature_points(quadrature_points),
      type(boundary_conditions.stacey.acoustic.type) {
  return;
}

template <typename property, typename qp_type>
template <specfem::enums::time_scheme::type time_scheme>
KOKKOS_INLINE_FUNCTION void specfem::enums::boundary_conditions::stacey<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic, property, qp_type>::
    mass_time_contribution(
        const int &ielement, const int &xz, const type_real &dt,
        const specfem::kokkos::array_type<type_real, 2> &weight,
        const specfem::compute::element_partial_derivatives
            &partial_derivatives,
        const specfem::compute::element_properties<
            specfem::enums::element::type::acoustic, property_type::value>
            &properties,
        specfem::kokkos::array_type<type_real, 1> &rmass_inverse) const {

  // Check if the GLL point is on the boundary
  // --------------------------------------------------------------------------

  constexpr int components = 1;
  int ngllx, ngllz;

  quadrature_points.get_ngll(&ngllx, &ngllz);

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);

  const auto itype = this->type(ielement);
  if (!specfem::compute::access::is_on_boundary(itype, iz, ix, ngllz, ngllx)) {
    return;
  }
  // --------------------------------------------------------------------------

  if constexpr (time_scheme == specfem::enums::time_scheme::type::newmark) {
    newmark_mass_terms(ix, iz, ngllx, ngllz, dt, itype, weight,
                       partial_derivatives, properties, rmass_inverse);
    return;
  }

  return;
}

template <typename property, typename qp_type>
KOKKOS_FUNCTION void specfem::enums::boundary_conditions::stacey<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic, property, qp_type>::
    enforce_traction(
        const int &ielement, const int &xz,
        const specfem::kokkos::array_type<type_real, 2> &weight,
        const specfem::compute::element_partial_derivatives
            &partial_derivatives,
        const specfem::compute::element_properties<
            specfem::enums::element::type::acoustic, property_type::value>
            &properties,
        const specfem::kokkos::array_type<type_real, 1> &field_dot,
        specfem::kokkos::array_type<type_real, 1> &field_dot_dot) const {

  // Check if the GLL point is on the boundary
  // --------------------------------------------------------------------------
  constexpr int components = 1;
  int ngllx, ngllz;

  quadrature_points.get_ngll(&ngllx, &ngllz);

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);

  const auto itype = this->type(ielement);
  if (!specfem::compute::access::is_on_boundary(itype, iz, ix, ngllz, ngllx)) {
    return;
  }
  // --------------------------------------------------------------------------

  // enforce traction condition
  // --------------------------------------------------------------------------
  // If the GLL point is on the corner the left or right traction conditions are
  // applied top or bottom traction conditions are ignored in this case. This
  // ensures there is no conflict in calculating the normal

  specfem::kokkos::array_type<type_real, dimension::dim> dn; // normal vector

  // Left Boundary
  if (itype.left && ix == 0) {
    dn = partial_derivatives
             .compute_normal<specfem::enums::boundaries::type::LEFT>();
    enforce_traction_boundary(weight[1], dn, properties, field_dot,
                              field_dot_dot);
    return;
  }

  // Right Boundary
  if (itype.right && ix == ngllx - 1) {
    dn = partial_derivatives
             .compute_normal<specfem::enums::boundaries::type::RIGHT>();
    enforce_traction_boundary(weight[1], dn, properties, field_dot,
                              field_dot_dot);
    return;
  }

  // Bottom Boundary
  if (itype.bottom && iz == 0) {
    dn = partial_derivatives
             .compute_normal<specfem::enums::boundaries::type::BOTTOM>();
    enforce_traction_boundary(weight[0], dn, properties, field_dot,
                              field_dot_dot);
    return;
  }

  // Top Boundary
  if (itype.top && iz == ngllz - 1) {
    dn = partial_derivatives
             .compute_normal<specfem::enums::boundaries::type::TOP>();
    enforce_traction_boundary(weight[0], dn, properties, field_dot,
                              field_dot_dot);
    return;
  }
  // --------------------------------------------------------------------------

  return;
}

#endif /* _ENUMS_BOUNDARY_CONDITIONS_STACEY_2D_ACOUSTIC_TPP_ */
