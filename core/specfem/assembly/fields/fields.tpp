#pragma once

#include "enumerations/interface.hpp"
#include "dim2/simulation_field.hpp"
#include "dim2/simulation_field.tpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"

// Explcitly instantiate the template class
template class specfem::assembly::simulation_field<specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::forward>;

template class specfem::assembly::simulation_field<specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::adjoint>;

template class specfem::assembly::simulation_field<specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::backward>;

template class specfem::assembly::simulation_field<specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::buffer>;

// Explicitly instantiate the sync_fields function for all combinations
template void specfem::assembly::simulation_field<specfem::dimension::type::dim2,
    specfem::wavefield::simulation_field::forward>::sync_fields<
    specfem::sync::kind::DeviceToHost>();
template void specfem::assembly::simulation_field<specfem::dimension::type::
    dim2, specfem::wavefield::simulation_field::forward>::sync_fields<
    specfem::sync::kind::HostToDevice>();
template void specfem::assembly::simulation_field<specfem::dimension::type::
    dim2, specfem::wavefield::simulation_field::adjoint>::sync_fields<
    specfem::sync::kind::DeviceToHost>();
template void specfem::assembly::simulation_field<specfem::dimension::type::
    dim2, specfem::wavefield::simulation_field::adjoint>::sync_fields
    <specfem::sync::kind::HostToDevice>();
template void specfem::assembly::simulation_field<specfem::dimension::type::
    dim2, specfem::wavefield::simulation_field::backward>::sync_fields<
    specfem::sync::kind::DeviceToHost>();
template void specfem::assembly::simulation_field<specfem::dimension::type::
    dim2, specfem::wavefield::simulation_field::backward>::sync_fields
    <specfem::sync::kind::HostToDevice>();
template void specfem::assembly::simulation_field<specfem::dimension::type::
    dim2, specfem::wavefield::simulation_field::buffer>::sync_fields<
    specfem::sync::kind::DeviceToHost>();
template void specfem::assembly::simulation_field<specfem::dimension::type::
    dim2, specfem::wavefield::simulation_field::buffer>::sync_fields
    <specfem::sync::kind::HostToDevice>();

specfem::assembly::fields<specfem::dimension::type::dim2>::fields(
    const specfem::assembly::mesh<dimension_tag> &mesh,
    const specfem::assembly::element_types<dimension_tag> &element_types,
    const specfem::simulation::type simulation)
    : // Initialize the forward field only if the simulation type is forward
      forward([&]() -> specfem::assembly::simulation_field<dimension_tag,
                        specfem::wavefield::simulation_field::forward> {
        if (simulation == specfem::simulation::type::forward) {
          return { mesh, element_types };
        } else if (simulation == specfem::simulation::type::combined) {
          return {};
        } else {
          throw std::runtime_error("Invalid simulation type");
        }
      }()),
      // Initiaze the adjoint field only if the simulation type is adjoint
      adjoint([&]() -> specfem::assembly::simulation_field<dimension_tag,
                        specfem::wavefield::simulation_field::adjoint> {
        if (simulation == specfem::simulation::type::forward) {
          return {};
        } else if (simulation == specfem::simulation::type::combined) {
          return { mesh, element_types };
        } else {
          throw std::runtime_error("Invalid simulation type");
        }
      }()),
      // Initialize the backward field only if the simulation type is adjoint
      backward([&]() -> specfem::assembly::simulation_field<dimension_tag,
                         specfem::wavefield::simulation_field::backward> {
        if (simulation == specfem::simulation::type::forward) {
          return {};
        } else if (simulation == specfem::simulation::type::combined) {
          return { mesh, element_types };
        } else {
          throw std::runtime_error("Invalid simulation type");
        }
      }()),
      // Initialize the buffer field only if the simulation type is adjoint
      buffer([&]() -> specfem::assembly::simulation_field<dimension_tag,
                       specfem::wavefield::simulation_field::buffer> {
        if (simulation == specfem::simulation::type::forward) {
          return { mesh, element_types };
        } else if (simulation == specfem::simulation::type::combined) {
          return { mesh, element_types };
        } else {
          throw std::runtime_error("Invalid simulation type");
        }
      }()) {}

void specfem::assembly::fields<specfem::dimension::type::dim2>::copy_to_device() {
  buffer.copy_to_device();
  forward.copy_to_device();
  adjoint.copy_to_device();
  backward.copy_to_device();
}

void specfem::assembly::fields<specfem::dimension::type::dim2>::copy_to_host() {
  buffer.copy_to_host();
  forward.copy_to_host();
  adjoint.copy_to_host();
  backward.copy_to_host();
}
