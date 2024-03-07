
#include "compute/fields/fields.hpp"
#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/impl/field_impl.tpp"
#include "compute/fields/simulation_field.hpp"
#include "compute/fields/simulation_field.tpp"

// Explcitly instantiate the template class
template class specfem::compute::simulation_field<
    specfem::wavefield::type::forward>;

template class specfem::compute::simulation_field<
    specfem::wavefield::type::adjoint>;

template class specfem::compute::simulation_field<
    specfem::wavefield::type::backward>;

specfem::compute::fields::fields(const specfem::compute::mesh &mesh,
                                 const specfem::compute::properties &properties,
                                 const specfem::simulation::type simulation)
    : // Initialize the forward field only if the simulation type is forward
      forward([&]() -> specfem::compute::simulation_field<
                        specfem::wavefield::type::forward> {
        if (simulation == specfem::simulation::type::forward) {
          return { mesh, properties };
        } else if (simulation == specfem::simulation::type::adjoint) {
          return {};
        } else {
          throw std::runtime_error("Invalid simulation type");
        }
      }()),
      // Initiaze the adjoint field only if the simulation type is adjoint
      adjoint([&]() -> specfem::compute::simulation_field<
                        specfem::wavefield::type::adjoint> {
        if (simulation == specfem::simulation::type::forward) {
          return {};
        } else if (simulation == specfem::simulation::type::adjoint) {
          return { mesh, properties };
        } else {
          throw std::runtime_error("Invalid simulation type");
        }
      }()),
      // Initialize the backward field only if the simulation type is adjoint
      backward([&]() -> specfem::compute::simulation_field<
                         specfem::wavefield::type::backward> {
        if (simulation == specfem::simulation::type::forward) {
          return {};
        } else if (simulation == specfem::simulation::type::adjoint) {
          return { mesh, properties };
        } else {
          throw std::runtime_error("Invalid simulation type");
        }
      }()) {}
