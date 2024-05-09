#ifndef _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELEMENT_KERNEL_HPP
#define _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELEMENT_KERNEL_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/kernels.hpp"
#include "point/properties.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace frechet_derivatives {
namespace impl {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
using AdjointPointFieldType =
    specfem::point::field<DimensionType, MediumTag, false, false, true, false>;

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
using BackwardPointFieldType =
    specfem::point::field<DimensionType, MediumTag, true, false, false, false>;

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
using PointFieldDerivativesType =
    specfem::point::field_derivatives<DimensionType, MediumTag>;

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::kernels<MediumTag, PropertyTag> element_kernel(
    const specfem::point::properties<MediumTag, PropertyTag> &properties,
    const AdjointPointFieldType<DimensionType, MediumTag> &adjoint_field,
    const BackwardPointFieldType<DimensionType, MediumTag> &backward_field,
    const PointFieldDerivativesType<DimensionType, MediumTag>
        &adjoint_derivatives,
    const PointFieldDerivativesType<DimensionType, MediumTag>
        &backward_derivatives,
    const type_real &dt);

} // namespace impl
} // namespace frechet_derivatives
} // namespace specfem

#endif /* _FRECHET_DERIVATIVES_IMPL_ELEMENT_KERNEL_ELEMENT_KERNEL_HPP */
