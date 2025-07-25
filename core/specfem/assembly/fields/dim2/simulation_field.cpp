#include "simulation_field.tpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/assembly/fields/impl/field_impl.tpp"

// Explicitly instantiate the template class
template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::forward>;

template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::adjoint>;

template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::backward>;

template class specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::buffer>;

template void specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::forward>::
    sync_fields<specfem::sync::kind::HostToDevice>();

template void specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::forward>::
    sync_fields<specfem::sync::kind::DeviceToHost>();

template void specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::adjoint>::
    sync_fields<specfem::sync::kind::HostToDevice>();

template void specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::adjoint>::
    sync_fields<specfem::sync::kind::DeviceToHost>();

template void specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::backward>::
    sync_fields<specfem::sync::kind::HostToDevice>();

template void specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::backward>::
    sync_fields<specfem::sync::kind::DeviceToHost>();

template void specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::buffer>::
    sync_fields<specfem::sync::kind::HostToDevice>();

template void specfem::assembly::simulation_field<
    specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::buffer>::
    sync_fields<specfem::sync::kind::DeviceToHost>();
