#ifndef _DOMAIN_SOURCE_ELASTIC_ISOTROPIC2D_TPP
#define _DOMAIN_SOURCE_ELASTIC_ISOTROPIC2D_TPP

#include "compute/interface.hpp"
#include "domain/impl/sources/elastic/elastic2d_isotropic.hpp"
#include "domain/impl/sources/source.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include "enumerations/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

// -----------------------------------------------------------------------------
//                     SPECIALIZED ELEMENT
// -----------------------------------------------------------------------------

template <int NGLL>
KOKKOS_FUNCTION specfem::domain::impl::sources::source<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    source(const specfem::compute::properties &properties,
           specfem::kokkos::DeviceView4d<type_real> source_array)
    : source_array(source_array) {

// #ifndef NDEBUG
//   assert(source_array.extent(1) == NGLL);
//   assert(source_array.extent(2) == NGLL);
// #endif

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::sources::source<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_interaction(const int &isource, const int &ispec, const int &xz,
                        const type_real &stf_value, type_real *acceleration) const {
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  static_assert(medium_type::components == 2,
                "Elastic medium must have 2 components");

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    acceleration[0] = source_array(isource, iz, ix, 0) * stf_value;
    acceleration[1] = source_array(isource, iz, ix, 1) * stf_value;
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    acceleration[0] = source_array(isource, iz, ix, 0) * stf_value;
    acceleration[1] = 0;
  }

  return;
}

#endif
