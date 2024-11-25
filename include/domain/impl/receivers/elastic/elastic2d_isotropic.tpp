#ifndef _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_TPP_
#define _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_TPP_

#include "compute/interface.hpp"
// #include "domain/impl/receivers/elastic/elastic2d.hpp"
#include "domain/impl/receivers/elastic/elastic2d_isotropic.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

// template <int NGLL>
// KOKKOS_FUNCTION specfem::domain::impl::receivers::receiver<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::elastic,
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
//       receiver_field(receiver_field) {}

template <int NGLL, bool using_simd>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::receivers::receiver<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    using_simd>::
    get_field(const int iz, const int ix,
              const specfem::point::partial_derivatives<
                  specfem::dimension::type::dim2, false, using_simd>
                  partial_derivatives,
              const specfem::point::properties<
                  specfem::dimension::type::dim2, medium_type::medium_tag,
                  medium_type::property_tag, using_simd>
                  properties,
              const ElementQuadratureViewType hprime,
              const ElementFieldType element_field,
              const specfem::enums::seismogram::type seismo_type,
              Kokkos::View<type_real[2], Kokkos::LayoutStride,
                           specfem::kokkos::DevMemSpace>
                  receiver_field) const {

  if(seismo_type == specfem::enums::seismogram::type::displacement
  || seismo_type == specfem::enums::seismogram::type::velocity
  || seismo_type == specfem::enums::seismogram::type::acceleration){

      const auto active_field = [&]() ->
        typename ElementFieldType::ViewType {
          switch (seismo_type) {
          case specfem::enums::seismogram::type::displacement:
            return element_field.displacement;
            break;
          case specfem::enums::seismogram::type::velocity:
            return element_field.velocity;
            break;
          case specfem::enums::seismogram::type::acceleration:
            return element_field.acceleration;
            break;
          default:
            DEVICE_ASSERT(false, "seismogram not supported");
            return {};
            break;
          }
        }();
    // Receiver field is probably not the best way of storing this, since this
    // would require global memory accesses. A better way for doing this would be
    // create register array and the store the values there. However, post
    // simulation people might require the field stored inside an element where a
    // receiver is located. If the number of receivers << nspec - hopefully this
    // shouldn't be a bottleneck.
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      receiver_field(0) = active_field(iz, ix, 0);
      receiver_field(1) = active_field(iz, ix, 1);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      receiver_field(0) = active_field(iz, ix, 0);
      receiver_field(1) = 0;
    }
  }else if(seismo_type == specfem::enums::seismogram::type::pressure){

    type_real dsx_dxi = 0.0;
    type_real dsx_dgamma = 0.0;
    type_real dsz_dxi = 0.0;
    type_real dsz_dgamma = 0.0;

#ifndef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int l = 0; l < NGLL; l++) {
      dsx_dxi += hprime(ix, l) * element_field.displacement(iz, l, 0);
      dsx_dgamma += hprime(iz, l) * element_field.displacement(l, ix, 0);
      dsz_dxi += hprime(ix, l) * element_field.displacement(iz, l, 1);
      dsz_dgamma += hprime(iz, l) * element_field.displacement(l, ix, 1);
    }
    type_real dsx_dx, dsx_dz, dsz_dx, dsz_dz;
    dsx_dx = dsx_dxi    * partial_derivatives.xix +
             dsx_dgamma * partial_derivatives.gammax;
    dsx_dz = dsx_dxi    * partial_derivatives.xiz +
             dsx_dgamma * partial_derivatives.gammaz;
    dsz_dx = dsz_dxi    * partial_derivatives.xix +
             dsz_dgamma * partial_derivatives.gammax;
    dsz_dz = dsz_dxi    * partial_derivatives.xiz +
             dsz_dgamma * partial_derivatives.gammaz;

    //https://specfem2d-kokkos.readthedocs.io/en/devel/api/datatypes/field_derivatives/point.html#_CPPv4N7specfem5point17field_derivatives2duE
    // tells us that fhe first index is the derivative index
#define du(i,j) (i==0? (j==0? dsx_dx:dsz_dx):(j==0? dsx_dz:dsz_dz))

    // WET code... should this be refactored with
    //   include/domain/impl/elements/elastic/elastic2d.tpp   ?

    type_real sigma_xx, sigma_zz, sigma_xz;
    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      // P_SV case
      // sigma_xx
      sigma_xx =
          properties.lambdaplus2mu * du(0, 0) + properties.lambda * du(1, 1);

      // sigma_zz
      sigma_zz =
          properties.lambdaplus2mu * du(1, 1) + properties.lambda * du(0, 0);

      // sigma_xz
      sigma_xz = properties.mu * (du(0, 1) + du(1, 0));
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      // SH-case
      // sigma_xx
      sigma_xx = properties.mu * du(0, 0); // would be sigma_xy in
                                          // CPU-version

      // sigma_xz
      sigma_xz = properties.mu * du(1, 0); // sigma_zy
    }

#undef du
    // https://github.com/SPECFEM/specfem2d/blob/98741db1a0c8082ca57364f5b17ea95df6cbf1c2/src/specfem2D/compute_pressure.f90#L398
    // pressure = - trace(sigma) / 3
    receiver_field(0) = - (sigma_xx + sigma_zz)/3.0;
    receiver_field(1) = 0;
  }else{
    DEVICE_ASSERT(false, "seismogram not supported");
  }

  return;
}

// template <int NGLL>
// KOKKOS_FUNCTION void specfem::domain::impl::receivers::receiver<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::elastic,
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
//     assert(false && "seismogram not supported");
//     break;
//   }
// }

// template <int NGLL>
// KOKKOS_FUNCTION void specfem::domain::impl::receivers::receiver<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::elastic,
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
//     receiver_seismogram(0) =
//         this->cos_rec(ireceiver) * seismogram_components[0] +
//         this->sin_rec(ireceiver) * seismogram_components[1];
//     receiver_seismogram(1) = 0;
//   }

//   return;
// }

#endif /* _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_TPP_ */
