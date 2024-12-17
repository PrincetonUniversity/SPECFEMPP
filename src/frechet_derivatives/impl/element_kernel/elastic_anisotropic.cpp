#include "frechet_derivatives/impl/element_kernel/elastic_anisotropic.hpp"
#include "frechet_derivatives/impl/element_kernel/elastic_anisotropic.tpp"

// Explicit template instantiation

// Specification for this template is:
// - UseSIMD = false
template KOKKOS_FUNCTION specfem::point::kernels<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::anisotropic, false>
specfem::frechet_derivatives::impl::impl_compute_element_kernel<false>(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, false> &properties,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::elastic, false,
                                false, true, false, false> &adjoint_field,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::elastic, true,
                                false, false, false, false> &backward_field,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        false> &adjoint_derivatives,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        false> &backward_derivatives,
    const type_real &dt);

// Specification for this template is:
// - UseSIMD = false
template KOKKOS_FUNCTION specfem::point::kernels<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::anisotropic, true>
specfem::frechet_derivatives::impl::impl_compute_element_kernel<true>(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::anisotropic, true> &properties,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::elastic, false,
                                false, true, false, true> &adjoint_field,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::elastic, true,
                                false, false, false, true> &backward_field,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        true> &adjoint_derivatives,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        true> &backward_derivatives,
    const type_real &dt);
