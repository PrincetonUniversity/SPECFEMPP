#include "compute/assembly/assembly.hpp"
#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"

specfem::compute::assembly::assembly(
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::quadrature::quadratures &quadratures,
    const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
    const std::vector<std::shared_ptr<specfem::receivers::receiver> >
        &receivers,
    const std::vector<specfem::enums::seismogram::type> &stypes,
    const type_real t0, const type_real dt, const int max_timesteps,
    const int max_sig_step, const specfem::simulation::type simulation,
    const bool has_gll_model) {
  this->mesh = { mesh.tags, mesh.control_nodes, quadratures };
  this->partial_derivatives = { this->mesh };
  this->properties = { this->mesh.nspec,   this->mesh.ngllz, this->mesh.ngllx,
                       this->mesh.mapping, mesh.tags,        mesh.materials,
                       has_gll_model };
  this->kernels = { this->mesh.nspec, this->mesh.ngllz, this->mesh.ngllx,
                    this->mesh.mapping, mesh.tags };
  this->sources = { sources,          this->mesh, this->partial_derivatives,
                    this->properties, t0,         dt,
                    max_timesteps };
  this->receivers = { max_sig_step, receivers, stypes, this->mesh };
  this->boundaries = { this->mesh.nspec,   this->mesh.ngllz,
                       this->mesh.ngllx,   mesh,
                       this->mesh.mapping, this->mesh.quadratures,
                       this->properties,   this->partial_derivatives };
  this->coupled_interfaces = { mesh,
                               this->mesh.points,
                               this->mesh.quadratures,
                               this->partial_derivatives,
                               this->properties,
                               this->mesh.mapping };
  this->fields = { this->mesh, this->properties, simulation };
  this->boundary_values = { max_timesteps, this->mesh, this->properties,
                            this->boundaries };
  return;
}
