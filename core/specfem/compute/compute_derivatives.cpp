#include "compute_derivatives.hpp"
#include "impl/compute_material_derivatives.hpp"
#include "impl/compute_material_derivatives.tpp"

// Explicit template instantiation
template void
specfem::compute::compute_derivatives<specfem::element::dimension_tag::dim2, 5>(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const type_real &);

template void
specfem::compute::compute_derivatives<specfem::element::dimension_tag::dim2, 8>(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const type_real &);
