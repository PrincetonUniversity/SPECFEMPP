#ifndef DOMAIN_ELEMENTS_HPP
#define DOMAIN_ELEMENTS_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress_integrand.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace elements {

// template <specfem::wavefield::type WavefieldType,
//           specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag,
//           specfem::element::boundary_tag BoundaryTag,
//           typename quadrature_points_type>
// class element;

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::stress_integrand<DimensionType, MediumTag>
compute_stress_integrands(
    const specfem::point::partial_derivatives2<false> &partial_derivatives,
    const specfem::point::properties<DimensionType, MediumTag, PropertyTag>
        &properties,
    const specfem::point::field_derivatives<DimensionType, MediumTag>
        &field_derivatives);

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION
    specfem::point::field<DimensionType, MediumTag, false, false, false, true>
    mass_matrix_component(
        const specfem::point::properties<DimensionType, MediumTag, PropertyTag>
            &properties,
        const specfem::point::partial_derivatives2<true> &partial_derivatives);

} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
