#ifndef _DOMAIN_SOURCE_ACOUSTIC_ISOTROPIC2D_TPP
#define _DOMAIN_SOURCE_ACOUSTIC_ISOTROPIC2D_TPP

#include <Kokkos_Core.hpp>
#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/acoustic2d_isotropic.hpp"
#include "domain/impl/sources/source.hpp"
#include "source_time_function/source_time_function.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"

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
    source(const int &ispec,
           const specfem::compute::properties &properties,
           specfem::kokkos::DeviceView3d<type_real> source_array,
           specfem::forcing_function::stf *stf)
    : ispec(ispec), stf(stf) {

#ifndef NDEBUG
  assert(source_array.extent(0) == NGLL);
  assert(source_array.extent(1) == NGLL);
  assert(properties.kappa.extent(1) == NGLL);
  assert(properties.kappa.extent(2) == NGLL);
#endif

  this->source_array = source_array;
  this->kappa = Kokkos::subview(properties.kappa, ispec, Kokkos::ALL(), Kokkos::ALL());

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::sources::source<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_interaction(const int &xz, const type_real &stf_value,
                        type_real *accel) const {
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  accel[0] = source_array(iz, ix, 0) * stf_value / kappa(iz, ix);

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::sources::source<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    update_acceleration(const type_real *accel,
                        field_type field_dot_dot) const {
  Kokkos::atomic_add(&field_dot_dot(0), accel[0]);

  return;
}

// -----------------------------------------------------------------------------

#endif
