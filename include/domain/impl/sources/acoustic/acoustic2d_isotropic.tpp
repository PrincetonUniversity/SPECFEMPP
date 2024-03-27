#ifndef _DOMAIN_SOURCE_ACOUSTIC_ISOTROPIC2D_TPP
#define _DOMAIN_SOURCE_ACOUSTIC_ISOTROPIC2D_TPP

#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/acoustic2d_isotropic.hpp"
#include "domain/impl/sources/source.hpp"
#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

using field_type = Kokkos::Subview<
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>, int,
    std::remove_const_t<decltype(Kokkos::ALL)> >;

// -----------------------------------------------------------------------------
//                     SPECIALIZED ELEMENT
// -----------------------------------------------------------------------------

// template <int NGLL>
// KOKKOS_FUNCTION specfem::domain::impl::sources::source<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::acoustic,
//     specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
//     specfem::enums::element::property::isotropic>::
//     source(const specfem::compute::properties &properties,
//            const specfem::kokkos::DeviceView4d<type_real> source_array)
//     : source_array(source_array), kappa(properties.kappa) {

//   // #ifndef NDEBUG
//   //   assert(source_array.extent(1) == NGLL);
//   //   assert(source_array.extent(2) == NGLL);
//   //   assert(properties.kappa.extent(1) == NGLL);
//   //   assert(properties.kappa.extent(2) == NGLL);
//   // #endif

//   return;
// }

template <int NGLL>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::sources::source<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL> >::
    compute_interaction(
        const specfem::kokkos::array_type<type_real, 1> &stf,
        const specfem::kokkos::array_type<type_real, 1> &lagrange_interpolant,
        specfem::kokkos::array_type<type_real, medium_type::components>
            &acceleration) const {

  acceleration[0] = lagrange_interpolant[0] * stf[0];

  return;
}

#endif
