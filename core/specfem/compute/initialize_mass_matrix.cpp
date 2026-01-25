#include "initialize_mass_matrix.hpp"
#include "enumerations/interface.hpp"
#include "impl/compute_mass_matrix.hpp"
#include "impl/compute_mass_matrix.tpp"
#include "impl/invert_mass_matrix.hpp"
#include "impl/invert_mass_matrix.tpp"

// Explicit template instantiation
// 2D, NGLL=5,8, forward, backward and adjoint wavefields
template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::forward, specfem::dimension::type::dim2,
    5>(specfem::assembly::assembly<specfem::dimension::type::dim2> &,
       const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::forward, specfem::dimension::type::dim2,
    8>(specfem::assembly::assembly<specfem::dimension::type::dim2> &,
       const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::backward, specfem::dimension::type::dim2,
    5>(specfem::assembly::assembly<specfem::dimension::type::dim2> &,
       const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::backward, specfem::dimension::type::dim2,
    8>(specfem::assembly::assembly<specfem::dimension::type::dim2> &,
       const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::adjoint, specfem::dimension::type::dim2,
    5>(specfem::assembly::assembly<specfem::dimension::type::dim2> &,
       const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::adjoint, specfem::dimension::type::dim2,
    8>(specfem::assembly::assembly<specfem::dimension::type::dim2> &,
       const type_real &);

// 3D, NGLL=5,8, forward, backward and adjoint wavefields
template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::forward, specfem::dimension::type::dim3,
    5>(specfem::assembly::assembly<specfem::dimension::type::dim3> &,
       const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::forward, specfem::dimension::type::dim3,
    8>(specfem::assembly::assembly<specfem::dimension::type::dim3> &,
       const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::backward, specfem::dimension::type::dim3,
    5>(specfem::assembly::assembly<specfem::dimension::type::dim3> &,
       const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::backward, specfem::dimension::type::dim3,
    8>(specfem::assembly::assembly<specfem::dimension::type::dim3> &,
       const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::adjoint, specfem::dimension::type::dim3,
    5>(specfem::assembly::assembly<specfem::dimension::type::dim3> &,
       const type_real &);

template void specfem::compute::initialize_mass_matrix<
    specfem::simulation::field_type::adjoint, specfem::dimension::type::dim3,
    8>(specfem::assembly::assembly<specfem::dimension::type::dim3> &,
       const type_real &);
