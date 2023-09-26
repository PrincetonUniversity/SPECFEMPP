#ifndef _DOMAIN_SOURCE_ACOUSTIC_ISOTROPIC2D_TPP
#define _DOMAIN_SOURCE_ACOUSTIC_ISOTROPIC2D_TPP

#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/acoustic2d_isotropic.hpp"
#include "domain/impl/sources/source.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

using field_type = Kokkos::Subview<
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>, int,
    std::remove_const_t<decltype(Kokkos::ALL)> >;

// -----------------------------------------------------------------------------
//                     SPECIALIZED ELEMENT
// -----------------------------------------------------------------------------

template <int NGLL>
KOKKOS_FUNCTION specfem::domain::impl::sources::source<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    source(const specfem::compute::properties &properties,
           const specfem::kokkos::DeviceView4d<type_real> source_array)
    : source_array(source_array), kappa(properties.kappa) {

#ifndef NDEBUG
  assert(source_array.extent(1) == NGLL);
  assert(source_array.extent(2) == NGLL);
  assert(properties.kappa.extent(1) == NGLL);
  assert(properties.kappa.extent(2) == NGLL);
#endif

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::sources::source<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_interaction(const int &isource, const int &ispec, const int &xz, const type_real &stf_value,
                        type_real *acceleration) const {
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  static_assert(medium_type::components == 1,
                "Acoustic medium must have 1 component");

  acceleration[0] = source_array(isource, iz, ix, 0) * stf_value / kappa(ispec, iz, ix);

  return;
}

#endif
