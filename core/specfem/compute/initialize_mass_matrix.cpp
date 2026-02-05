#include "initialize_mass_matrix.hpp"
#include "enumerations/interface_tags.hpp"
#include "impl/compute_mass_matrix.hpp"
#include "impl/compute_mass_matrix.tpp"
#include "impl/invert_mass_matrix.hpp"
#include "impl/invert_mass_matrix.tpp"

// Explicit template instantiation
// 2D, NGLL=5,8, forward, backward and adjoint wavefields
template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::forward,
    specfem::element::dimension_tag::dim2, 5>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::forward,
    specfem::element::dimension_tag::dim2, 8>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::backward,
    specfem::element::dimension_tag::dim2, 5>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::backward,
    specfem::element::dimension_tag::dim2, 8>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::adjoint,
    specfem::element::dimension_tag::dim2, 5>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::adjoint,
    specfem::element::dimension_tag::dim2, 8>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const type_real &);

// 3D, NGLL=5,8, forward, backward and adjoint wavefields
template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::forward,
    specfem::element::dimension_tag::dim3, 5>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::forward,
    specfem::element::dimension_tag::dim3, 8>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::backward,
    specfem::element::dimension_tag::dim3, 5>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::backward,
    specfem::element::dimension_tag::dim3, 8>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::adjoint,
    specfem::element::dimension_tag::dim3, 5>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::adjoint,
    specfem::element::dimension_tag::dim3, 8>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const type_real &);
