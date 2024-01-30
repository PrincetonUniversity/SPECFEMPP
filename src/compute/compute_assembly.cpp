
#include "compute/compute_assembly.hpp"
#include "compute/boundaries/boundaries.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/compute_receivers.hpp"
#include "compute/compute_sources.hpp"
#include "compute/coupled_interfaces/coupled_interfaces.hpp"
#include "compute/fields/fields.hpp"
#include "compute/properties/interface.hpp"
#include "mesh/mesh.hpp"

specfem::compute::assembly::assembly(
    const specfem::mesh::mesh &mesh,
    const specfem::quadrature::quadratures &quadratures,
    const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
    const std::vector<std::shared_ptr<specfem::receivers::receiver> >
        &receivers,
    const std::vector<specfem::enums::seismogram::type> &stypes,
    const int max_sig_step) {
  this->mesh = specfem::compute::mesh(mesh.control_nodes, quadratures);
  this->partial_derivatives = specfem::compute::partial_derivatives(this->mesh);
  this->properties = specfem::compute::properties(
      this->mesh.nspec, this->mesh.ngllz, this->mesh.ngllx, mesh.materials);
  this->sources =
      specfem::compute::sources(sources, this->mesh, this->partial_derivatives,
                                this->properties, max_sig_step);
  this->receivers =
      specfem::compute::receivers(max_sig_step, receivers, stypes, this->mesh);
  this->boundaries =
      specfem::compute::boundaries(this->mesh.nspec, this->properties,
                                   mesh.abs_boundary, mesh.acfree_surface);
  this->coupled_interfaces = specfem::compute::coupled_interfaces(
      this->mesh, this->properties, mesh.coupled_interfaces);
  this->fields = specfem::compute::fields(this->mesh, this->properties);
  return;
}
