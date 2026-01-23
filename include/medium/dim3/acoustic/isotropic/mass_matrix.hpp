#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium {

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION
    specfem::point::mass_inverse<specfem::dimension::type::dim3,
                                 specfem::element::medium_tag::acoustic,
                                 UseSIMD>
    impl_mass_matrix_component(
        const specfem::point::properties<specfem::dimension::type::dim3,
                                         specfem::element::medium_tag::acoustic,
                                         PropertyTag, UseSIMD> &properties);

} // namespace specfem::medium
