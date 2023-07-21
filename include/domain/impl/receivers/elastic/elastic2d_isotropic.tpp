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

using value_type = type_real[];

template <int NGLL>
specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    receiver(const int irec, const int iseis, const int ispec,
             const type_real sin_rec, const type_real cos_rec,
             const specfem::enums::seismogram::type seismogram,
             const specfem::compute::receivers receivers,
             const specfem::kokkos::DeviceView3d<int> ibool,
             specfem::quadrature::quadrature *gllx,
             specfem::quadrature::quadrature *gllz)
    : ispec(ispec), sin_rec(sin_rec), cos_rec(cos_rec), gllx(gllx), gllz(gllz) {
  this->receiver_field = specfem::kokkos::DeviceView3d<type_real>(
      "receiver_field", specfem::enums::element::dimension::dim2::dim, NGLL,
      NGLL);
  this->receiver_array = Kokkos::subview(receivers.receiver_array, irec,
                                         Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  // Subview the seismogram array at current receiver and current seismogram
  // value
  this->receiver_seismogram = Kokkos::subview(receivers.seismogram, Kokkos::ALL,
                                              iseis, irec, Kokkos::ALL);
  this->ibool = Kokkos::subview(ibool, ispec, Kokkos::ALL, Kokkos::ALL);

  return;
}

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    get_field(
        const int xz, const specfem::kokkos::DeviceView2d<type_real> field,
        const specfem::kokkos::DeviceView2d<type_real> field_dot,
        const specfem::kokkos::DeviceView2d<type_real> field_dot_dot) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);
  const int iglob = ibool(iz, ix);

  switch (this->seismogram) {
  case specfem::enums::seismogram::type::displacement:
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      receiver_field(0, iz, ix) = field(iglob, 0);
      receiver_field(1, iz, ix) = field(iglob, 1);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      receiver_field(0, iz, ix) = field(iglob, 0);
    }
    break;
  case specfem::enums::seismogram::type::velocity:
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      receiver_field(0, iz, ix) = field_dot(iglob, 0);
      receiver_field(1, iz, ix) = field_dot(iglob, 1);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      receiver_field(0, iz, ix) = field_dot(iglob, 0);
    }
    break;
  case specfem::enums::seismogram::type::acceleration:
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      receiver_field(0, iz, ix) = field_dot_dot(iglob, 0);
      receiver_field(1, iz, ix) = field_dot_dot(iglob, 1);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      receiver_field(0, iz, ix) = field_dot_dot(iglob, 0);
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
    compute_seismogram_components(const int xz,
                                  type_real (&l_seismogram_components)[2]) const {
  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  switch (this->seismogram) {
  case specfem::enums::seismogram::type::displacement:
  case specfem::enums::seismogram::type::velocity:
  case specfem::enums::seismogram::type::acceleration:
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      const type_real hlagrange = this->receiver_array(iz, ix, 0);
      const type_real field_v = receiver_field(0, iz, ix);
      l_seismogram_components[0] +=
          this->receiver_array(iz, ix, 0) * this->receiver_field(0, iz, ix);
      l_seismogram_components[1] +=
          this->receiver_array(iz, ix, 1) * this->receiver_field(1, iz, ix);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      l_seismogram_components[0] +=
          this->receiver_array(iz, ix, 0) * this->receiver_field(0, iz, ix);
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
    compute_seismogram(const int isig_step,
                       const type_real (&seismogram_components)[2]) const {

  assert(seismogram_components.extent(1) == 2);
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
