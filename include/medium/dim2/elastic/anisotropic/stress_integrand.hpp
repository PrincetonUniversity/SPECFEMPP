#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "globals.h"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress_integrand.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium {

// Elastic 2D isotropic specialization
// stress_integrand = \sum_{i,k=1}^{2} F_{ik} \partial_i w^{\alpha\gamma}
template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
    UseSIMD>
impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<
        specfem::dimension::type::dim2, false, UseSIMD> &partial_derivatives,
    const specfem::point::properties<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_sv,
        specfem::element::property_tag::anisotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_sv, UseSIMD> &field_derivatives);

} // namespace medium
} // namespace specfem
