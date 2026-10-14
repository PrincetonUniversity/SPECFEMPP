#include "specfem/assembly/boundaries.hpp"

#include "specfem/assembly/boundaries/dim3/impl/injection/initialize_injection.hpp"

std::shared_ptr<specfem::injection::injection_provider<
    specfem::element::dimension_tag::dim3>>
specfem::assembly::boundaries<specfem::element::dimension_tag::dim3>::
    initialize_injection(
        const specfem::injection::injection_inputs &inputs,
        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            &mesh_assembly,
        const specfem::assembly::properties<
            specfem::element::dimension_tag::dim3> &properties,
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim3> &element_types,
        type_real dt, int nstep) {
  return specfem::assembly::boundaries_impl::make_injection_boundary(
      inputs, this->stacey, this->stacey_index_mapping,
      this->h_stacey_index_mapping, mesh_assembly, properties, element_types,
      dt, nstep, this->injection);
}
