
#include "compute/fields/fields.hpp"
#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/impl/field_impl.tpp"
#include "compute/fields/simulation_field.hpp"
#include "compute/fields/simulation_field.tpp"

specfem::compute::fields::fields(const specfem::compute::mesh &mesh,
                                 const specfem::compute::properties &properties)
    : forward(mesh, properties) {}

// Explcitly instantiate the template class
template class specfem::compute::simulation_field<
    specfem::enums::simulation::forward>;
