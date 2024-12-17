#pragma once

#include "compute/interface.hpp"
#include "domain/impl/sources/elastic/elastic2d_anisotropic.hpp"
#include "domain/impl/sources/source.hpp"
#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

// -----------------------------------------------------------------------------
//                     SPECIALIZED ELEMENT
// -----------------------------------------------------------------------------

// template <int NGLL>
// KOKKOS_FUNCTION specfem::domain::impl::sources::source<
//     specfem::enums::element::dimension::dim2,
//     specfem::enums::element::medium::elastic,
//     specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
//     specfem::enums::element::property::isotropic>::
//     source(const specfem::compute::properties &properties,
//            specfem::kokkos::DeviceView4d<type_real> source_array) {

//   // #ifndef NDEBUG
//   //   assert(source_array.extent(1) == NGLL);
//   //   assert(source_array.extent(2) == NGLL);
//   // #endif

//   return;
// }

template <int NGLL, bool using_simd>
KOKKOS_INLINE_FUNCTION void specfem::domain::impl::sources::source<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::anisotropic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    using_simd>::
    compute_interaction(
        const specfem::datatype::ScalarPointViewType<
            type_real, components, using_simd> &stf,
        const specfem::datatype::ScalarPointViewType<
            type_real, components, using_simd>
            &lagrange_interpolant,
        specfem::datatype::ScalarPointViewType<
            type_real, components, using_simd> &acceleration)
        const {

  if constexpr (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    acceleration(0) = lagrange_interpolant(0) * stf(0);
    acceleration(1) = lagrange_interpolant(1) * stf(1);
  } else if constexpr (specfem::globals::simulation_wave == specfem::wave::sh) {
    acceleration(0) = lagrange_interpolant(0) * stf(0);
    acceleration(1) = 0;
  }

  return;
}
