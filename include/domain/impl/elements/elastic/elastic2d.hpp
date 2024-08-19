#ifndef _DOMAIN_ELASTIC_ELEMENTS2D_HPP
#define _DOMAIN_ELASTIC_ELEMENTS2D_HPP

#include "../element.hpp"
#include "domain/impl/elements/element.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "globals.h"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress_integrand.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace elements {

// Elastic 2D isotropic specialization
// stress_integrand = \sum_{i,k=1}^{2} F_{ik} \partial_i w^{\alpha\gamma}
template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    UseSIMD>
impl_compute_stress_integrands(
    const specfem::point::partial_derivatives2<UseSIMD, false>
        &partial_derivatives,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        UseSIMD> &field_derivatives);

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::elastic,
                                      false, false, false, true, UseSIMD>
impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::partial_derivatives2<UseSIMD, true>
        &partial_derivatives);
} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
