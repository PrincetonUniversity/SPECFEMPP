#ifndef DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTROPIC_TPP_
#define DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTROPIC_TPP_

#include "constants.hpp"
#include "domain/impl/receivers/acoustic/acoustic2d.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

template <int NGLL>
KOKKOS_FUNCTION specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    receiver(const specfem::kokkos::DeviceView1d<type_real> sin_rec,
             const specfem::kokkos::DeviceView1d<type_real> cos_rec,
             const specfem::kokkos::DeviceView4d<type_real> receiver_array,
             const specfem::compute::partial_derivatives &partial_derivatives,
             const specfem::compute::properties &properties,
             specfem::kokkos::DeviceView6d<type_real> receiver_field)
    : sin_rec(sin_rec), cos_rec(cos_rec), receiver_array(receiver_array),
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

  this->xix = partial_derivatives.xix;
  this->gammax = partial_derivatives.gammax;
  this->xiz = partial_derivatives.xiz;
  this->gammaz = partial_derivatives.gammaz;
  this->rho_inverse = properties.rho_inverse;
  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    get_field(
        const int &ireceiver, const int &iseis, const int &ispec,
        const specfem::enums::seismogram::type &seismogram_type, const int &xz,
        const int &isig_step,
        const ScratchViewType<type_real, medium_type::components> field,
        const ScratchViewType<type_real, medium_type::components> field_dot,
        const ScratchViewType<type_real, medium_type::components> field_dot_dot,
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

  const type_real xixl = this->xix(ispec, iz, ix);
  const type_real gammaxl = this->gammax(ispec, iz, ix);
  const type_real xizl = this->xiz(ispec, iz, ix);
  const type_real gammazl = this->gammaz(ispec, iz, ix);
  const type_real rho_inversel = this->rho_inverse(ispec, iz, ix);

  using sv_ScratchViewType =
      Kokkos::Subview<ScratchViewType<type_real, medium_type::components>,
                      std::remove_const_t<decltype(Kokkos::ALL)>,
                      std::remove_const_t<decltype(Kokkos::ALL)>, int>;

  sv_ScratchViewType active_field;

  switch (seismogram_type) {
  case specfem::enums::seismogram::type::displacement:
    active_field = Kokkos::subview(field, Kokkos::ALL, Kokkos::ALL, 0);
    break;
  case specfem::enums::seismogram::type::velocity:
    active_field = Kokkos::subview(field_dot, Kokkos::ALL, Kokkos::ALL, 0);
    break;
  case specfem::enums::seismogram::type::acceleration:
    active_field = Kokkos::subview(field_dot_dot, Kokkos::ALL, Kokkos::ALL, 0);
    break;
  }

  type_real dchi_dxi = 0.0;
  type_real dchi_dgamma = 0.0;

#ifndef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; l++) {
    dchi_dxi += hprime_xx(ix, l, 0) * active_field(iz, l);
    dchi_dgamma += hprime_zz(iz, l, 0) * active_field(l, ix);
  }

  // dchidx
  type_real fieldx = (dchi_dxi * xixl + dchi_dgamma * gammaxl) * rho_inversel;

  // dchidz
  type_real fieldz = (dchi_dxi * xizl + dchi_dgamma * gammazl) * rho_inversel;

  // Receiver field is probably not the best way of storing this, since this
  // would require global memory accesses. A better way for doing this would be
  // create register array and the store the values there. However, post
  // simulation people might require the field stored inside an element where a
  // receiver is located. If the number of receivers << nspec - hopefully this
  // shouldn't be a bottleneck.
  this->receiver_field(isig_step, ireceiver, iseis, 0, iz, ix) = fieldx;
  this->receiver_field(isig_step, ireceiver, iseis, 1, iz, ix) = fieldz;

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_seismogram_components(
        const int &ireceiver, const int &iseis,
        const specfem::enums::seismogram::type &seismogram_type, const int &xz,
        const int &isig_step,
        specfem::kokkos::array_type<type_real, 2> &l_seismogram_components) const {
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  switch (seismogram_type) {
  case specfem::enums::seismogram::type::displacement:
  case specfem::enums::seismogram::type::velocity:
  case specfem::enums::seismogram::type::acceleration:
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      l_seismogram_components[0] +=
          this->receiver_array(ireceiver, iz, ix, 0) *
          this->receiver_field(isig_step, ireceiver, iseis, 0, iz, ix);
      l_seismogram_components[1] +=
          this->receiver_array(ireceiver, iz, ix, 1) *
          this->receiver_field(isig_step, ireceiver, iseis, 1, iz, ix);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      l_seismogram_components[0] +=
          this->receiver_array(ireceiver, iz, ix, 0) *
          this->receiver_field(isig_step, ireceiver, iseis, 0, iz, ix);
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
        const int &ireceiver,
        const specfem::kokkos::array_type<type_real, 2> &seismogram_components,
        specfem::kokkos::DeviceView1d<type_real> receiver_seismogram) const {

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    receiver_seismogram(0) =
        this->cos_rec(ireceiver) * seismogram_components[0] +
        this->sin_rec(ireceiver) * seismogram_components[1];
    receiver_seismogram(1) =
        this->sin_rec(ireceiver) * seismogram_components[0] +
        this->cos_rec(ireceiver) * seismogram_components[1];
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    receiver_seismogram(0) =
        this->cos_rec(ireceiver) * seismogram_components[0] +
        this->sin_rec(ireceiver) * seismogram_components[1];
    receiver_seismogram(1) = 0;
  }

  return;
}

#endif /* DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTROPIC_TPP_ */
