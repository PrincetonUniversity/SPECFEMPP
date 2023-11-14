#ifndef _ENUMS_BOUNDARY_CONDITIONS_STACEY_2D_ACOUSTIC_TPP_
#define _ENUMS_BOUNDARY_CONDITIONS_STACEY_2D_ACOUSTIC_TPP_

#include "compute/interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "stacey2d_acoustic.hpp"

template <typename qp_type>
specfem::enums::boundary_conditions::stacey<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    qp_type>::stacey(const specfem::compute::boundaries &boundary_conditions,
                     const quadrature_points_type &quadrature_points)
    : quadrature_points(quadrature_points),
      type(boundary_conditions.stacey.acoustic.type) {
  return;
}

KOKKOS_INLINE_FUNCTION static void enforce_traction_boundary(
    const type_real &weight,
    const specfem::kokkos::array_type<type_real, 2> &dn,
    const specfem::compute::element_properties<
        specfem::enums::element::type::acoustic,
        specfem::enums::element::property_tag::isotropic> &properties,
    const specfem::kokkos::array_type<type_real, 1> &field_dot,
    specfem::kokkos::array_type<type_real, 1> &field_dot_dot) {

  field_dot_dot[0] =
      // jacobian1d * weight * rho_vp_inverse * velocity
      dn.l2_norm() * weight * properties.rho_vpinverse * field_dot[0];
  return;
}

template <typename qp_type>
template <specfem::enums::element::property_tag property>
KOKKOS_FUNCTION void specfem::enums::boundary_conditions::stacey<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic, qp_type>::
    enforce_traction(
        const int &ielement, const int &xz,
        const specfem::kokkos::array_type<type_real, 2> &weight,
        const specfem::compute::element_partial_derivatives
            &partial_derivatives,
        const specfem::compute::element_properties<
            specfem::enums::element::type::acoustic, property> &properties,
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
    dn = partial_derivatives.compute_normal<specfem::enums::boundaries::type::LEFT>();
    enforce_traction_boundary(weight[1], dn, properties, field_dot, field_dot_dot);
    return;
  }

  // Right Boundary
  if (itype.right && ix == ngllx - 1) {
    dn = partial_derivatives.compute_normal<specfem::enums::boundaries::type::RIGHT>();
    enforce_traction_boundary(weight[1], dn, properties, field_dot, field_dot_dot);
    return;
  }

  // Bottom Boundary
  if (itype.bottom && iz == 0) {
    dn = partial_derivatives.compute_normal<specfem::enums::boundaries::type::BOTTOM>();
    enforce_traction_boundary(weight[0], dn, properties, field_dot, field_dot_dot);
    return;
  }

  // Top Boundary
  if (itype.top && iz == ngllz - 1) {
    dn = partial_derivatives.compute_normal<specfem::enums::boundaries::type::TOP>();
    enforce_traction_boundary(weight[0], dn, properties, field_dot, field_dot_dot);
    return;
  }
  // --------------------------------------------------------------------------

  return;
}

#endif /* _ENUMS_BOUNDARY_CONDITIONS_STACEY_2D_ACOUSTIC_TPP_ */
