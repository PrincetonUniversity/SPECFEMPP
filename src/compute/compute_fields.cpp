
#include "compute/fields/fields.hpp"
#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/impl/field_impl.tpp"
#include "compute/fields/simulation_field.hpp"
#include "compute/fields/simulation_field.tpp"

// Explcitly instantiate the template class
template class specfem::compute::simulation_field<
    specfem::enums::simulation::type::forward>;

template class specfem::compute::simulation_field<
    specfem::enums::simulation::type::adjoint>;

specfem::compute::fields::fields(
    const specfem::compute::mesh &mesh,
    const specfem::compute::properties &properties,
    const specfem::enums::simulation::type simulation)
    : forward(mesh, properties),
      // Initiaze the adjoint field only if the simulation type is adjoint
      // Otherwise the views are not allocated. It is upto the solver not to use
      // the adjoint field if the simulation type is forward
      adjoint([&]() -> specfem::compute::simulation_field<
                        specfem::enums::simulation::type::adjoint> {
        if (simulation == specfem::enums::simulation::type::forward) {
          return {};
        } else if (simulation == specfem::enums::simulation::type::adjoint) {
          return { mesh, properties };
        } else {
          throw std::runtime_error("Invalid simulation type");
        }
      }()) {}
