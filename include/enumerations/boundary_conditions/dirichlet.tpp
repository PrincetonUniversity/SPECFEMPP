#ifndef _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_TPP_
#define _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_TPP_

#include "compute/interface.hpp"
#include "dirichlet.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>

template <typename dim, typename medium, typename qp_type>
specfem::enums::boundary_conditions::dirichlet<dim, medium, qp_type>::dirichlet(
    const specfem::compute::boundaries &boundary_conditions,
    const quadrature_points_type &quadrature_points)
    : quadrature_points(quadrature_points),
      type(boundary_conditions.acoustic_free_surface.type) {
  return;
}

template <typename dim, typename medium, typename qp_type>
template <specfem::enums::element::property_tag property>
KOKKOS_FUNCTION void specfem::enums::boundary_conditions::
    dirichlet<dim, medium, qp_type>::enforce_traction(
        const int &ielement, const int &xz,
        const specfem::kokkos::array_type<type_real, dimension::dim>
            &weight,
        const specfem::compute::element_partial_derivatives
            &partial_derivatives,
        const specfem::compute::element_properties<medium_type::value, property>
            &properties,
        const specfem::kokkos::array_type<type_real, medium_type::components>
            &field_dot,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &field_dot_dot) const {

  constexpr int components = medium_type::components;
  int ngllx, ngllz;
  quadrature_points.get_ngll(&ngllx, &ngllz);

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);

  const auto itype = this->type(ielement);
  if (!specfem::compute::access::is_on_boundary(itype, iz, ix, ngllz, ngllx)) {
    return;
  }

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int icomp = 0; icomp < components; ++icomp)
    field_dot_dot[icomp] = 0.0;

  return;
}

#endif /* _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_TPP_ */
