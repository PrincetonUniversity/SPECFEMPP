#ifndef _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_TPP_
#define _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_TPP_

#include "compute/interface.hpp"
#include "domain/impl/receivers/elastic/elastic2d.hpp"
#include "domain/impl/receivers/elastic/elastic2d_isotropic.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
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
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    receiver(const type_real sin_rec, const type_real cos_rec,
             const specfem::enums::seismogram::type seismogram,
             const sv_receiver_array_type receiver_array,
             const sv_receiver_seismogram_type receiver_seismogram,
             const specfem::kokkos::DeviceView2d<int> ibool,
             const sv_receiver_field_type receiver_field)
    : sin_rec(sin_rec), cos_rec(cos_rec), seismogram(seismogram),
      receiver_seismogram(receiver_seismogram), receiver_array(receiver_array),
      ibool(ibool), receiver_field(receiver_field) {}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    get_field(const int xz, const int isig_step,
              const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
                  field,
              const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
                  field_dot,
              const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
                  field_dot_dot) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);
  const int iglob = ibool(iz, ix);

  switch (this->seismogram) {
  case specfem::enums::seismogram::type::displacement:
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      this->receiver_field(isig_step, 0, iz, ix) = field(iglob, 0);
      this->receiver_field(isig_step, 1, iz, ix) = field(iglob, 1);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      this->receiver_field(isig_step, 0, iz, ix) = field(iglob, 0);
    }
    break;
  case specfem::enums::seismogram::type::velocity:
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      this->receiver_field(isig_step, 0, iz, ix) = field_dot(iglob, 0);
      this->receiver_field(isig_step, 1, iz, ix) = field_dot(iglob, 1);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      this->receiver_field(isig_step, 0, iz, ix) = field_dot(iglob, 0);
    }
    break;
  case specfem::enums::seismogram::type::acceleration:
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      this->receiver_field(isig_step, 0, iz, ix) = field_dot_dot(iglob, 0);
      this->receiver_field(isig_step, 1, iz, ix) = field_dot_dot(iglob, 1);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      this->receiver_field(isig_step, 0, iz, ix) = field_dot_dot(iglob, 0);
    }
    break;
  default:
    // seismogram not supported
    assert(false);
    break;
  }

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
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
      l_seismogram_components[0] +=
          this->receiver_array(iz, ix, 0) * this->receiver_field(isig_step, 0, iz, ix);
      l_seismogram_components[1] +=
          this->receiver_array(iz, ix, 1) * this->receiver_field(isig_step, 1, iz, ix);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      l_seismogram_components[0] +=
          this->receiver_array(iz, ix, 0) * this->receiver_field(isig_step, 0, iz, ix);
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
    specfem::enums::element::medium::elastic,
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

#endif /* _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_TPP_ */
