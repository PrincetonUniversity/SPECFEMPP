#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium {

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::field<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv_t,
    false, false, false, true, UseSIMD>
impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_psv_t,
        specfem::element::property_tag::isotropic_cosserat, UseSIMD>
        &properties,
    const specfem::point::partial_derivatives<
        specfem::dimension::type::dim2, true, UseSIMD> &partial_derivatives);

} // namespace medium
} // namespace specfem
