#include "fields.hpp"
#include "fields/simulation_field.hpp"
#include "fields/simulation_field.tpp"

// Explcitly instantiate the template class
template class specfem::assembly::simulation_field<
    specfem::wavefield::simulation_field::forward>;

template class specfem::assembly::simulation_field<
    specfem::wavefield::simulation_field::adjoint>;

template class specfem::assembly::simulation_field<
    specfem::wavefield::simulation_field::backward>;

template class specfem::assembly::simulation_field<
    specfem::wavefield::simulation_field::buffer>;

specfem::assembly::fields::fields(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    const specfem::simulation::type simulation)
    : // Initialize the forward field only if the simulation type is forward
      forward([&]() -> specfem::assembly::simulation_field<
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
      adjoint([&]() -> specfem::assembly::simulation_field<
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
      backward([&]() -> specfem::assembly::simulation_field<
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
      buffer([&]() -> specfem::assembly::simulation_field<
                       specfem::wavefield::simulation_field::buffer> {
        if (simulation == specfem::simulation::type::forward) {
          return { mesh, element_types };
        } else if (simulation == specfem::simulation::type::combined) {
          return { mesh, element_types };
        } else {
          throw std::runtime_error("Invalid simulation type");
        }
      }()) {}
