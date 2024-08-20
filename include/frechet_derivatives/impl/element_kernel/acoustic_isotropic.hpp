#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/kernels.hpp"
#include "point/properties.hpp"

namespace specfem {
namespace frechet_derivatives {
namespace impl {

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::kernels<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, UseSIMD>
impl_compute_element_kernel(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::acoustic, false,
                                false, true, false, UseSIMD> &adjoint_field,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::acoustic, true,
                                false, false, false, UseSIMD> &backward_field,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        UseSIMD> &adjoint_derivatives,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        UseSIMD> &backward_derivatives,
    const type_real &dt);

} // namespace impl
} // namespace frechet_derivatives
} // namespace specfem
