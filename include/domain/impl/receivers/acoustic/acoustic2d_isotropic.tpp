#ifndef DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTROPIC_TPP_
#define DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTROPIC_TPP_

#include "constants.hpp"
// #include "domain/impl/receivers/acoustic/acoustic2d.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

// template <int NGLL>
// KOKKOS_FUNCTION specfem::domain::impl::receivers::receiver<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::acoustic,
//     specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
//     specfem::enums::element::property::isotropic>::
//     receiver(const specfem::kokkos::DeviceView1d<type_real> sin_rec,
//              const specfem::kokkos::DeviceView1d<type_real> cos_rec,
//              const specfem::kokkos::DeviceView4d<type_real> receiver_array,
//              const specfem::compute::partial_derivatives
//              &partial_derivatives, const specfem::compute::properties
//              &properties, specfem::kokkos::DeviceView6d<type_real>
//              receiver_field)
//     : sin_rec(sin_rec), cos_rec(cos_rec), receiver_array(receiver_array),
//       receiver_field(receiver_field) {

// #ifndef NDEBUG
//   assert(partial_derivatives.xix.extent(1) == NGLL);
//   assert(partial_derivatives.xix.extent(2) == NGLL);
//   assert(partial_derivatives.gammax.extent(1) == NGLL);
//   assert(partial_derivatives.gammax.extent(2) == NGLL);
//   assert(partial_derivatives.xiz.extent(1) == NGLL);
//   assert(partial_derivatives.xiz.extent(2) == NGLL);
//   assert(partial_derivatives.gammaz.extent(1) == NGLL);
//   assert(partial_derivatives.gammaz.extent(2) == NGLL);

//   // Properties
//   assert(properties.rho_inverse.extent(1) == NGLL);
//   assert(properties.rho_inverse.extent(2) == NGLL);
// #endif

//   this->xix = partial_derivatives.xix;
//   this->gammax = partial_derivatives.gammax;
//   this->xiz = partial_derivatives.xiz;
//   this->gammaz = partial_derivatives.gammaz;
//   this->rho_inverse = properties.rho_inverse;
//   return;
// }

template <int NGLL, bool using_simd>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>, using_simd>::
    get_field(
        const int iz, const int ix,
        const specfem::point::partial_derivatives<specfem::dimension::type::dim2, false, using_simd>
            partial_derivatives,
        const specfem::point::properties<specfem::dimension::type::dim2, medium_type::medium_tag,
                                         medium_type::property_tag, using_simd>
            properties,
        const ElementQuadratureViewType hprime,
        const ElementFieldViewType active_field,
        Kokkos::View<type_real[2], Kokkos::LayoutStride,
                     specfem::kokkos::DevMemSpace>
            receiver_field) const {

  type_real dchi_dxi = 0.0;
  type_real dchi_dgamma = 0.0;

#ifndef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; l++) {
    dchi_dxi += hprime(ix, l) * active_field(iz, l, 0);
    dchi_dgamma += hprime(iz, l) * active_field(l, ix, 0);
  }

  // dchidx
  type_real fieldx = (dchi_dxi * partial_derivatives.xix +
                      dchi_dgamma * partial_derivatives.gammax);

  // dchidz
  type_real fieldz = (dchi_dxi * partial_derivatives.xiz +
                      dchi_dgamma * partial_derivatives.gammaz);

  // Receiver field is probably not the best way of storing this, since this
  // would require global memory accesses. A better way for doing this would be
  // create register array and the store the values there. However, post
  // simulation people might require the field stored inside an element where a
  // receiver is located. If the number of receivers << nspec - hopefully this
  // shouldn't be a bottleneck.
  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    receiver_field(0) = fieldx * properties.rho_inverse;
    receiver_field(1) = fieldz * properties.rho_inverse;
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    receiver_field(0) = fieldx * properties.rho_inverse;
    receiver_field(1) = 0;
  }

  return;
}

// template <int NGLL>
// KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::acoustic,
//     specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
//     specfem::enums::element::property::isotropic>::
//     compute_seismogram_components(
//         const int &ireceiver, const int &iseis,
//         const specfem::enums::seismogram::type &seismogram_type, const int
//         &xz, const int &isig_step, specfem::kokkos::array_type<type_real, 2>
//         &l_seismogram_components) const {
//   int ix, iz;
//   sub2ind(xz, NGLL, iz, ix);

//   switch (seismogram_type) {
//   case specfem::enums::seismogram::type::displacement:
//   case specfem::enums::seismogram::type::velocity:
//   case specfem::enums::seismogram::type::acceleration:
//     if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
//       l_seismogram_components[0] +=
//           this->receiver_array(ireceiver, iz, ix, 0) *
//           this->receiver_field(isig_step, ireceiver, iseis, 0, iz, ix);
//       l_seismogram_components[1] +=
//           this->receiver_array(ireceiver, iz, ix, 1) *
//           this->receiver_field(isig_step, ireceiver, iseis, 1, iz, ix);
//     } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
//       l_seismogram_components[0] +=
//           this->receiver_array(ireceiver, iz, ix, 0) *
//           this->receiver_field(isig_step, ireceiver, iseis, 0, iz, ix);
//       l_seismogram_components[1] += 0;
//     }
//     break;

//   default:
//     // seismogram not supported
//     assert(false);
//     break;
//   }
// }

// template <int NGLL>
// KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::acoustic,
//     specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
//     specfem::enums::element::property::isotropic>::
//     compute_seismogram(
//         const int &ireceiver,
//         const specfem::kokkos::array_type<type_real, 2>
//         &seismogram_components, specfem::kokkos::DeviceView1d<type_real>
//         receiver_seismogram) const {

//   if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
//     receiver_seismogram(0) =
//         this->cos_rec(ireceiver) * seismogram_components[0] +
//         this->sin_rec(ireceiver) * seismogram_components[1];
//     receiver_seismogram(1) =
//         this->sin_rec(ireceiver) * seismogram_components[0] +
//         this->cos_rec(ireceiver) * seismogram_components[1];
//   } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
//     receiver_seismogram(0) = seismogram_components[0];
//     receiver_seismogram(1) = 0;
//   }

//   return;
// }

#endif /* DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTROPIC_TPP_ */
