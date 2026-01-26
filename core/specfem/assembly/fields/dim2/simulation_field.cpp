#include "simulation_field.tpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/field_impl.tpp"

// Explicitly instantiate the template class
template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim2, specfem::simulation::field_type::forward>;

template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim2, specfem::simulation::field_type::adjoint>;

template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim2, specfem::simulation::field_type::backward>;

template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim2, specfem::simulation::field_type::buffer>;

template void
specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                    specfem::simulation::field_type::forward>::
    sync_fields<specfem::sync::kind::HostToDevice>();

template void
specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                    specfem::simulation::field_type::forward>::
    sync_fields<specfem::sync::kind::DeviceToHost>();

template void
specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                    specfem::simulation::field_type::adjoint>::
    sync_fields<specfem::sync::kind::HostToDevice>();

template void
specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                    specfem::simulation::field_type::adjoint>::
    sync_fields<specfem::sync::kind::DeviceToHost>();

template void
specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                    specfem::simulation::field_type::backward>::
    sync_fields<specfem::sync::kind::HostToDevice>();

template void
specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                    specfem::simulation::field_type::backward>::
    sync_fields<specfem::sync::kind::DeviceToHost>();

template void
specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                    specfem::simulation::field_type::buffer>::
    sync_fields<specfem::sync::kind::HostToDevice>();

template void
specfem::assembly::simulation_field<specfem::dimension::type::dim2,
                                    specfem::simulation::field_type::buffer>::
    sync_fields<specfem::sync::kind::DeviceToHost>();
