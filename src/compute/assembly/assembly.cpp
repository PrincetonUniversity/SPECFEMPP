#include "compute/assembly/assembly.hpp"
#include "enumerations/interface.hpp"
#include "io/reader.hpp"
#include "mesh/mesh.hpp"

specfem::compute::assembly::assembly(
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::quadrature::quadratures &quadratures,
    const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
    const std::vector<std::shared_ptr<specfem::receivers::receiver> >
        &receivers,
    const std::vector<specfem::enums::seismogram::type> &stypes,
    const type_real t0, const type_real dt, const int max_timesteps,
    const int max_sig_step, const int nsteps_between_samples,
    const specfem::simulation::type simulation,
    const std::shared_ptr<specfem::io::reader> &property_reader) {
  this->mesh = { mesh.tags, mesh.control_nodes, quadratures };
  this->element_types = { this->mesh.nspec, this->mesh.ngllz, this->mesh.ngllx,
                          this->mesh.mapping, mesh.tags };
  this->partial_derivatives = { this->mesh };
  this->properties = { this->mesh.nspec, this->mesh.ngllz,
                       this->mesh.ngllx, this->element_types,
                       mesh.materials,   property_reader != nullptr };
  this->kernels = { this->mesh.nspec, this->mesh.ngllz, this->mesh.ngllx,
                    this->element_types };
  this->sources = {
    sources, this->mesh,   this->partial_derivatives, this->element_types, t0,
    dt,      max_timesteps
  };
  this->receivers = { this->mesh.nspec,
                      this->mesh.ngllz,
                      this->mesh.ngllz,
                      max_sig_step,
                      dt,
                      t0,
                      nsteps_between_samples,
                      receivers,
                      stypes,
                      this->mesh,
                      mesh.tags,
                      this->element_types };
  this->boundaries = { this->mesh.nspec,   this->mesh.ngllz,
                       this->mesh.ngllx,   mesh,
                       this->mesh.mapping, this->mesh.quadratures,
                       this->properties,   this->partial_derivatives };
  this->coupled_interfaces = { mesh,
                               this->mesh.points,
                               this->mesh.quadratures,
                               this->partial_derivatives,
                               this->element_types,
                               this->mesh.mapping };
  this->fields = { this->mesh, this->element_types, simulation };
  this->boundary_values = { max_timesteps, this->mesh, this->element_types,
                            this->boundaries };

  /// Add some domain checks here for SH domains
  const int nelastic_sh = this->element_types.get_number_of_elements(
      specfem::element::medium_tag::elastic_sh);

  const int nacoustic = this->element_types.get_number_of_elements(
      specfem::element::medium_tag::acoustic);

  if (nelastic_sh > 0 && nacoustic > 0) {
    std::ostringstream msg;
    msg << "Elastic SH and acoustic elements cannot be mixed in the same "
        << "domain. We currently do not support SH and pressure wave coupling. "
        << "Please check your MESHFEM input file.";

    throw std::runtime_error(msg.str());
  }
  return;
}
