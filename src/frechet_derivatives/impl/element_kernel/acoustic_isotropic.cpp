#include "frechet_derivatives/impl/element_kernel/acoustic_isotropic.hpp"
#include "frechet_derivatives/impl/element_kernel/acoustic_isotropic.tpp"

// Explicit template instantiation
template KOKKOS_FUNCTION specfem::point::kernels<
    specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, false>
specfem::frechet_derivatives::impl::impl_compute_element_kernel<false>(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, false> &properties,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::acoustic, false,
                                false, true, false, false> &adjoint_field,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::acoustic, true,
                                false, false, false, false> &backward_field,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        false> &adjoint_derivatives,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        false> &backward_derivatives,
    const type_real &dt);

template KOKKOS_FUNCTION specfem::point::kernels<
    specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic, true>
specfem::frechet_derivatives::impl::impl_compute_element_kernel<true>(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, true> &properties,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::acoustic, false,
                                false, true, false, true> &adjoint_field,
    const specfem::point::field<specfem::dimension::type::dim2,
                                specfem::element::medium_tag::acoustic, true,
                                false, false, false, true> &backward_field,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        true> &adjoint_derivatives,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        true> &backward_derivatives,
    const type_real &dt);
