#ifndef _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_TPP_
#define _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_TPP_

#include "compute/interface.hpp"
#include "domain/impl/receivers/elastic/elastic2d.hpp"
#include "domain/impl/receivers/elastic/elastic2d_isotropic.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <int NGLL>
KOKKOS_FUNCTION specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    receiver(const specfem::kokkos::DeviceView1d<type_real> sin_rec,
             const specfem::kokkos::DeviceView1d<type_real> cos_rec,
             const specfem::kokkos::DeviceView4d<type_real> receiver_array,
             const specfem::compute::partial_derivatives &partial_derivatives,
             const specfem::compute::properties &properties,
             specfem::kokkos::DeviceView6d<type_real> receiver_field)
    : sin_rec(sin_rec), cos_rec(cos_rec), receiver_array(receiver_array),
      receiver_field(receiver_field) {}

template <int NGLL>
KOKKOS_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    get_field(
        const int &ireceiver, const int &iseis, const int &ispec,
        const specfem::enums::seismogram::type &seismogram_type, const int &xz,
        const int &isig_step,
        const ScratchViewType<type_real, medium_type::components> field,
        const ScratchViewType<type_real, medium_type::components> field_dot,
        const ScratchViewType<type_real, medium_type::components> field_dot_dot,
        const ScratchViewType<type_real, 1> s_hprime_xx,
        const ScratchViewType<type_real, 1> s_hprime_zz) const {

#ifndef NDEBUG
  // check that the dimensions of the fields are correct
  assert(field.extent(0) == NGLL);
  assert(field.extent(1) == NGLL);
  assert(field_dot.extent(0) == NGLL);
  assert(field_dot.extent(1) == NGLL);
  assert(field_dot_dot.extent(0) == NGLL);
  assert(field_dot_dot.extent(1) == NGLL);
  assert(s_hprime_xx.extent(0) == NGLL);
  assert(s_hprime_xx.extent(1) == NGLL);
  assert(s_hprime_zz.extent(0) == NGLL);
  assert(s_hprime_zz.extent(1) == NGLL);
#endif /* NDEBUG */

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  using sv_ScratchViewType =
      Kokkos::Subview<ScratchViewType<type_real, medium_type::components>,
                      std::remove_const_t<decltype(Kokkos::ALL)>,
                      std::remove_const_t<decltype(Kokkos::ALL)>, int>;

  sv_ScratchViewType active_fieldx;
  sv_ScratchViewType active_fieldz;

  switch (seismogram_type) {
  case specfem::enums::seismogram::type::displacement:
    active_fieldx = Kokkos::subview(field, Kokkos::ALL, Kokkos::ALL, 0);
    active_fieldz = Kokkos::subview(field, Kokkos::ALL, Kokkos::ALL, 1);
    break;
  case specfem::enums::seismogram::type::velocity:
    active_fieldx = Kokkos::subview(field_dot, Kokkos::ALL, Kokkos::ALL, 0);
    active_fieldz = Kokkos::subview(field_dot, Kokkos::ALL, Kokkos::ALL, 1);
    break;
  case specfem::enums::seismogram::type::acceleration:
    active_fieldx = Kokkos::subview(field_dot_dot, Kokkos::ALL, Kokkos::ALL, 0);
    active_fieldz = Kokkos::subview(field_dot_dot, Kokkos::ALL, Kokkos::ALL, 1);
    break;
  default:
    // seismogram not supported
    assert(false && "seismogram not supported");
    break;
  }

  // Receiver field is probably not the best way of storing this, since this
  // would require global memory accesses. A better way for doing this would be
  // create register array and the store the values there. However, post
  // simulation people might require the field stored inside an element where a
  // receiver is located. If the number of receivers << nspec - hopefully this
  // shouldn't be a bottleneck.
  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    this->receiver_field(isig_step, ireceiver, iseis, 0, iz, ix) =
        active_fieldx(iz, ix);
    this->receiver_field(isig_step, ireceiver, iseis, 1, iz, ix) =
        active_fieldz(iz, ix);
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    this->receiver_field(isig_step, ireceiver, iseis, 0, iz, ix) =
        active_fieldx(iz, ix);
  }

  return;
}

template <int NGLL>
KOKKOS_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>::
    compute_seismogram_components(
        const int &ireceiver, const int &iseis,
        const specfem::enums::seismogram::type &seismogram_type, const int &xz,
        const int &isig_step,
        specfem::kokkos::array_type<type_real, 2> &l_seismogram_components)
        const {
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
    assert(false && "seismogram not supported");
    break;
  }
}

template <int NGLL>
KOKKOS_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
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

#endif /* _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_TPP_ */
