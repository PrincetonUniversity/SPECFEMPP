#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "globals.h"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium {

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
    UseSIMD>
impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_psv,
                                     PropertyTag, UseSIMD> &properties);

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sh,
    UseSIMD>
impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sh,
                                     PropertyTag, UseSIMD> &properties);

} // namespace medium
} // namespace specfem
