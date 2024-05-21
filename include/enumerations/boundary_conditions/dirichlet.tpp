#ifndef _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_TPP_
#define _ENUMS_BOUNDARY_CONDITIONS_DIRICHLET_TPP_

#include "compute/interface.hpp"
#include "dirichlet.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include <Kokkos_Core.hpp>

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename qp_type>
KOKKOS_INLINE_FUNCTION void specfem::boundary::boundary<
    WavefieldType, specfem::dimension::type::dim2, MediumTag, PropertyTag,
    specfem::element::boundary_tag::acoustic_free_surface, qp_type>::
    enforce_traction(
        const int &xz,
        const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
        const specfem::point::partial_derivatives2<true> &partial_derivatives,
        const specfem::point::properties<medium_type::medium_tag,
                                         medium_type::property_tag> &properties,
        const specfem::point::boundary &boundary_type,
        const specfem::kokkos::array_type<type_real, medium_type::components>
            &field_dot,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &field_dot_dot) const {

  constexpr int components = medium_type::components;
  constexpr auto value_t = value;

  int ngllx, ngllz;
  quadrature_points.get_ngll(&ngllx, &ngllz);

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);

  if (!specfem::point::is_on_boundary(value_t, boundary_type, iz, ix, ngllz,
                                      ngllx)) {
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
