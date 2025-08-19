#include "enumerations/interface.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/field_impl.tpp"

// Explicitly instantiate the template class
template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim3,
    specfem::wavefield::simulation_field::forward>;

template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim3,
    specfem::wavefield::simulation_field::adjoint>;

template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim3,
    specfem::wavefield::simulation_field::backward>;

template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim3,
    specfem::wavefield::simulation_field::buffer>;
