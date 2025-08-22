#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium {

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    UseSIMD>
impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties);

} // namespace medium
} // namespace specfem
