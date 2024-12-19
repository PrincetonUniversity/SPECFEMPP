#include "compute/kernels/kernels.hpp"

// Explicit template instantiation

template class specfem::compute::impl::kernels::material_kernels<
    specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic>;

template class specfem::compute::impl::kernels::material_kernels<
    specfem::element::medium_tag::acoustic,
    specfem::element::property_tag::isotropic>;

