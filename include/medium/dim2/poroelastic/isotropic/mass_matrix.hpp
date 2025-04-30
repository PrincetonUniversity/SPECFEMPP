#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/point/field.hpp"
#include "specfem/point/partial_derivatives.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium {

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::poroelastic,
                                      false, false, false, true, UseSIMD>
impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::poroelastic,
                                     specfem::element::property_tag::isotropic,
                                     UseSIMD> &properties,
    const specfem::point::partial_derivatives<
        specfem::dimension::type::dim2, true, UseSIMD> &partial_derivatives);

} // namespace medium
} // namespace specfem
