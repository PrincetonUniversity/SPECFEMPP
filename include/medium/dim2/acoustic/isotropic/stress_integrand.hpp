#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress_integrand.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium {

// Acoustic 2D isotropic specialization
// stress_integrand = rho^{-1} * \sum_{i,k=1}^{2} \partial_i w^{\alpha\gamma}
// \partial_i \chi
template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    UseSIMD>
impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<
        specfem::dimension::type::dim2, false, UseSIMD> &partial_derivatives,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        UseSIMD> &field_derivatives);

} // namespace medium
} // namespace specfem
