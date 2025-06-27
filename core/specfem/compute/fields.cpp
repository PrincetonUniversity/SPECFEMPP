
#include "compute/fields/fields.hpp"
#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/impl/field_impl.tpp"
#include "compute/fields/simulation_field.hpp"
#include "compute/fields/simulation_field.tpp"

// Explcitly instantiate the template class
template class specfem::compute::simulation_field<
    specfem::wavefield::simulation_field::forward>;

template class specfem::compute::simulation_field<
    specfem::wavefield::simulation_field::adjoint>;

template class specfem::compute::simulation_field<
    specfem::wavefield::simulation_field::backward>;

template class specfem::compute::simulation_field<
    specfem::wavefield::simulation_field::buffer>;

specfem::compute::fields::fields(
    const specfem::compute::mesh &mesh,
    const specfem::compute::element_types &element_types,
    const specfem::simulation::type simulation)
    : // Initialize the forward field only if the simulation type is forward
      forward([&]() -> specfem::compute::simulation_field<
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
      adjoint([&]() -> specfem::compute::simulation_field<
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
      backward([&]() -> specfem::compute::simulation_field<
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
      buffer([&]() -> specfem::compute::simulation_field<
                       specfem::wavefield::simulation_field::buffer> {
        if (simulation == specfem::simulation::type::forward) {
          return { mesh, element_types };
        } else if (simulation == specfem::simulation::type::combined) {
          return { mesh, element_types };
        } else {
          throw std::runtime_error("Invalid simulation type");
        }
      }()) {}
