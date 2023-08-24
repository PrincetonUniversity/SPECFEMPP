#ifndef DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTROPIC_TPP_
#define DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTROPIC_TPP_

#include "constants.hpp"
#include "domain/impl/receivers/acoustic/acoustic2d.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include <Kokkos_Core.hpp>

using sv_receiver_array_type =
    Kokkos::Subview<specfem::kokkos::DeviceView4d<type_real>, int,
                    std::remove_const_t<decltype(Kokkos::ALL)>,
                    std::remove_const_t<decltype(Kokkos::ALL)>,
                    std::remove_const_t<decltype(Kokkos::ALL)> >;

using sv_receiver_seismogram_type =
    Kokkos::Subview<specfem::kokkos::DeviceView4d<type_real>,
                    std::remove_const_t<decltype(Kokkos::ALL)>, int, int,
                    std::remove_const_t<decltype(Kokkos::ALL)> >;

using sv_receiver_field_type =
    Kokkos::Subview<specfem::kokkos::DeviceView6d<type_real>,
                    std::remove_const_t<decltype(Kokkos::ALL)>, int, int,
                    std::remove_const_t<decltype(Kokkos::ALL)>,
                    std::remove_const_t<decltype(Kokkos::ALL)>,
                    std::remove_const_t<decltype(Kokkos::ALL)> >;

template <int NGLL>
KOKKOS_FUNCTION specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic >::
    receiver(const int ispec, const type_real sin_rec, const type_real cos_rec,
             const specfem::enums::seismogram::type seismogram,
             const sv_receiver_array_type receiver_array,
             const sv_receiver_seismogram_type receiver_seismogram,
             const specfem::compute::partial_derivatives &partial_derivatives,
             const specfem::compute::properties &properties,
             sv_receiver_field_type receiver_field)
    : ispec(ispec), sin_rec(sin_rec), cos_rec(cos_rec), seismogram(seismogram),
      receiver_seismogram(receiver_seismogram), receiver_array(receiver_array),
      receiver_field(receiver_field) {

#ifndef NDEBUG
  assert(partial_derivatives.xix.extent(1) == NGLL);
  assert(partial_derivatives.xix.extent(2) == NGLL);
  assert(partial_derivatives.gammax.extent(1) == NGLL);
  assert(partial_derivatives.gammax.extent(2) == NGLL);
  assert(partial_derivatives.xiz.extent(1) == NGLL);
  assert(partial_derivatives.xiz.extent(2) == NGLL);
  assert(partial_derivatives.gammaz.extent(1) == NGLL);
  assert(partial_derivatives.gammaz.extent(2) == NGLL);

  // Properties
  assert(properties.rho_inverse.extent(1) == NGLL);
  assert(properties.rho_inverse.extent(2) == NGLL);
#endif

  this->xix = Kokkos::subview(partial_derivatives.xix, ispec, Kokkos::ALL(),
                              Kokkos::ALL());
  this->gammax = Kokkos::subview(partial_derivatives.gammax, ispec,
                                 Kokkos::ALL(), Kokkos::ALL());
  this->xiz = Kokkos::subview(partial_derivatives.xiz, ispec, Kokkos::ALL(),
                              Kokkos::ALL());
  this->gammaz = Kokkos::subview(partial_derivatives.gammaz, ispec,
                                 Kokkos::ALL(), Kokkos::ALL());
  this->rho_inverse = Kokkos::subview(properties.rho_inverse, ispec,
                                      Kokkos::ALL(), Kokkos::ALL());
  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    get_field(const int xz, const int isig_step,
              const ScratchViewType<type_real, medium::components> field,
              const ScratchViewType<type_real, medium::components> field_dot,
              const ScratchViewType<type_real, medium::components> field_dot_dot,
              const ScratchViewType<type_real, 1> hprime_xx,
              const ScratchViewType<type_real, 1> hprime_zz) const {

#ifndef NDEBUG
  assert(field.extent(0) == NGLL);
  assert(field.extent(1) == NGLL);
  assert(field_dot.extent(0) == NGLL);
  assert(field_dot.extent(1) == NGLL);
  assert(field_dot_dot.extent(0) == NGLL);
  assert(field_dot_dot.extent(1) == NGLL);
  assert(hprime_xx.extent(0) == NGLL);
  assert(hprime_xx.extent(1) == NGLL);
  assert(hprime_zz.extent(0) == NGLL);
  assert(hprime_zz.extent(1) == NGLL);
#endif

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real xixl = this->xix(iz, ix);
  const type_real gammaxl = this->gammax(iz, ix);
  const type_real xizl = this->xiz(iz, ix);
  const type_real gammazl = this->gammaz(iz, ix);
  const type_real rho_inversel = this->rho_inverse(iz, ix);

  ScratchViewType<type_real, 1> active_field;

  switch (this->seismogram) {
  case specfem::enums::seismogram::type::displacement:
    active_field = field;
    break;
  case specfem::enums::seismogram::type::velocity:
    active_field = field_dot;
    break;
  case specfem::enums::seismogram::type::acceleration:
    active_field = field_dot_dot;
    break;
  }

  type_real dchi_dxi = 0.0;
  type_real dchi_dgamma = 0.0;

  for (int l = 0; l < NGLL; l++) {
    dchi_dxi += hprime_xx(ix, l) * active_field(iz, l);
    dchi_dgamma += hprime_zz(iz, l) * active_field(l, ix);
  }

  // dchidx
  type_real fieldx = (dchi_dxi * xixl + dchi_dgamma * gammaxl) * rho_inversel;

  // dchidz
  type_real fieldz = (dchi_dxi * xizl + dchi_dgamma * gammazl) * rho_inversel;

  this->receiver_field(isig_step, 0, iz, ix) = fieldx;
  this->receiver_field(isig_step, 1, iz, ix) = fieldz;

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_seismogram_components(
        const int xz, const int isig_step,
        dimension::array_type<type_real> &l_seismogram_components) const {
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  switch (this->seismogram) {
  case specfem::enums::seismogram::type::displacement:
  case specfem::enums::seismogram::type::velocity:
  case specfem::enums::seismogram::type::acceleration:
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      l_seismogram_components[0] += this->receiver_array(iz, ix, 0) *
                                    this->receiver_field(isig_step, 0, iz, ix);
      l_seismogram_components[1] += this->receiver_array(iz, ix, 1) *
                                    this->receiver_field(isig_step, 1, iz, ix);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      l_seismogram_components[0] += this->receiver_array(iz, ix, 0) *
                                    this->receiver_field(isig_step, 0, iz, ix);
      l_seismogram_components[1] += 0;
    }
    break;

  default:
    // seismogram not supported
    assert(false);
    break;
  }
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_seismogram(
        const int isig_step,
        const dimension::array_type<type_real> &seismogram_components) {

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    receiver_seismogram(isig_step, 0) =
        this->cos_rec * seismogram_components[0] +
        this->sin_rec * seismogram_components[1];
    receiver_seismogram(isig_step, 1) =
        this->sin_rec * seismogram_components[0] +
        this->cos_rec * seismogram_components[1];
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    receiver_seismogram(isig_step, 0) =
        this->cos_rec * seismogram_components[0] +
        this->sin_rec * seismogram_components[1];
    receiver_seismogram(isig_step, 1) = 0;
  }

  return;
}

#endif /* DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTROPIC_TPP_ */
