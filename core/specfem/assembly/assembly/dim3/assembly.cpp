#include "assembly.hpp"
#include "enumerations/interface.hpp"
#include "io/reader.hpp"
#include "specfem/mesh.hpp"

specfem::assembly::assembly<specfem::dimension::type::dim3>::assembly(
    const specfem::mesh::mesh<dimension_tag> &mesh,
    const specfem::quadrature::quadratures &quadratures,
    std::vector<std::shared_ptr<specfem::sources::source<dimension_tag> > >
        &sources,
    const std::vector<
        std::shared_ptr<specfem::receivers::receiver<dimension_tag> > >
        &receivers,
    const std::vector<specfem::wavefield::type> &stypes, const type_real t0,
    const type_real dt, const int max_timesteps, const int max_sig_step,
    const int nsteps_between_samples,
    const specfem::simulation::type simulation,
    const bool allocate_boundary_values,
    const std::shared_ptr<specfem::io::reader> &property_reader) {

  const int nspec = mesh.nspec;
  const int ngllz = mesh.element_grid.ngllz;
  const int nglly = mesh.element_grid.nglly;
  const int ngllx = mesh.element_grid.ngllx;
  const int ngnod = mesh.control_nodes.ngnod;

  this->mesh = { nspec,
                 ngnod,
                 ngllz,
                 nglly,
                 ngllx,
                 mesh.adjacency_graph,
                 mesh.control_nodes,
                 quadratures };

  this->element_types = { nspec, ngllz, nglly, ngllx, this->mesh, mesh.tags };

  this->jacobian_matrix = { this->mesh };

  this->properties = { nspec, ngllz,          nglly,
                       ngllx, mesh.materials, this->element_types };

  this->kernels = { this->mesh.nspec, this->mesh.element_grid.ngllz,
                    this->mesh.element_grid.nglly,
                    this->mesh.element_grid.ngllx, this->element_types };

  this->sources = {
    sources, this->mesh, this->jacobian_matrix, this->element_types,
    t0,      dt,         max_timesteps
  };
  this->receivers = {
    max_sig_step, dt,         t0,        nsteps_between_samples, receivers,
    stypes,       this->mesh, mesh.tags, this->element_types
  };
  this->boundaries = { this->mesh.nspec,
                       this->mesh.element_grid.ngllz,
                       this->mesh.element_grid.nglly,
                       this->mesh.element_grid.ngllx,
                       mesh,
                       this->mesh,
                       this->jacobian_matrix };
  // this->coupled_interfaces = { mesh, this->mesh, this->jacobian_matrix,
  //                              this->element_types };
  this->fields = { this->mesh, this->element_types, simulation };

  // if (allocate_boundary_values)
  //   this->boundary_values = { max_timesteps, this->mesh, this->element_types,
  //                             this->boundaries };

  // Currently done in the mesher!
  this->check_jacobian_matrix();

  return;
}

std::string
specfem::assembly::assembly<specfem::dimension::type::dim3>::print() const {
  std::ostringstream message;
  message << "Assembly information:\n"
          << "------------------------------\n"
          << "  Total number of spectral elements             : "
          << this->mesh.nspec << "\n"
          << "  Total number of quadrature points per element : "
          << this->mesh.element_grid.ngllz << "\n";
  // << "Total number of distinct quadrature points    : "
  // << this->mesh.nglob << "\n";

  int total_elements = 0;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC)), {
        // Getting the number of elements per medium
        int n_elements = this->element_types.get_number_of_elements(
            _medium_tag_, _property_tag_);

        // Printing the number of elements if more than 0
        if (n_elements > 0) {
          // Adding the number of elements to the total
          total_elements += n_elements;

          message << "   Total number of elements of type "
                  << specfem::element::to_string(_medium_tag_, _property_tag_)
                  << " : " << n_elements << "\n";
        };
      })

  if (total_elements == mesh.nspec) {
    message << "  All elements accounted for.\n";
  } else {
    message << " NOT ALL ELEMENTS ACCOUNTED FOR\n";
    message << "  Mesh elements:              " << mesh.nspec << "\n";
    message << "  Assembly elements counted:  " << total_elements << "\n";
    message << "  Total unaccounted elements: " << (mesh.nspec - total_elements)
            << "\n";
    throw std::runtime_error(message.str());
  }
  return message.str();
}
