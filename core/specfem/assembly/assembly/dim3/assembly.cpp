#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/tag_dispatch/for_each.hpp"

specfem::assembly::assembly<specfem::element::dimension_tag::dim3>::assembly(
    const specfem::mesh::mesh<dimension_tag> &mesh,
    const specfem::quadrature::quadratures &quadratures,
    std::vector<std::shared_ptr<specfem::sources::source<dimension_tag>>>
        &sources,
    const std::vector<
        std::shared_ptr<specfem::receivers::receiver<dimension_tag>>>
        &receivers,
    const std::vector<specfem::enums::wavefield> &stypes, const type_real t0,
    const type_real dt, const int max_timesteps, const int max_sig_step,
    const int nsteps_between_samples,
    const specfem::simulation::type simulation,
    const bool allocate_boundary_values,
    const std::shared_ptr<specfem::io::reader> &property_reader,
    const specfem::element_coupling::flux_scheme_configuration
        &flux_scheme_config) {

  this->t0 = t0;
  this->dt = dt;

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
                 mesh.tags,
                 mesh.adjacency_graph,
                 mesh.control_nodes,
                 quadratures };

  this->element_types = { nspec, this->mesh.element_grid, this->mesh,
                          mesh.tags };

  this->element_intersections = { ngllz, nglly, ngllx, this->mesh,
                                  this->element_types };

  this->jacobian_matrix = { this->mesh };

  this->field_derivative_storage = { this->element_types, this->mesh.nspec,
                                     this->mesh.element_grid.ngllz,
                                     this->mesh.element_grid.nglly,
                                     this->mesh.element_grid.ngllx };

  this->kernels = { this->element_types };

  this->sources = {
    sources, this->mesh,   mesh, this->jacobian_matrix, this->element_types, t0,
    dt,      max_timesteps
  };
  this->receivers = {
    max_sig_step, dt,         t0,   nsteps_between_samples, receivers,
    stypes,       this->mesh, mesh, this->element_types
  };
  this->boundaries = { this->mesh.nspec,
                       this->mesh.element_grid.ngllz,
                       this->mesh.element_grid.nglly,
                       this->mesh.element_grid.ngllx,
                       mesh,
                       this->mesh,
                       this->jacobian_matrix };
  this->conforming_interfaces = {
    ngllz,     nglly, ngllx, this->element_intersections, this->jacobian_matrix,
    this->mesh
  };
  this->fields = { this->mesh, this->element_types, simulation };

  this->mpi_interfaces = { mesh.adjacency_graph,
                           this->element_types,
                           simulation,
                           this->fields,
                           this->mesh.h_mesh_to_compute,
                           ngllz,
                           nglly,
                           ngllx };

  if (allocate_boundary_values)
    this->boundary_values = { max_timesteps, this->mesh, this->element_types,
                              this->boundaries };

  // Currently done in the mesher!
  this->check_jacobian_matrix();

  this->attenuation = { mesh.attenuation, dt, this->mesh, this->element_types,
                        mesh.materials };

  this->properties = { this->element_types, this->mesh, mesh.materials,
                       property_reader != nullptr };

  this->info = { this->mesh, this->properties, this->element_types };

  return;
}

std::string
specfem::assembly::assembly<specfem::element::dimension_tag::dim3>::print()
    const {
  std::ostringstream message;
  int nspec = this->mesh.nspec;
  const auto comm = specfem::MPI::communicator();
  SPECFEM_MPI_SAFECALL(
      MPI_Allreduce(MPI_IN_PLACE, &nspec, 1, MPI_INT, MPI_SUM, comm));

  message << "\nAssembly information:\n"
          << "---------------------\n\n"
          << "Total number of spectral elements             : " << nspec << "\n"
          << "Total number of quadrature points per element : "
          << this->mesh.element_grid.ngllz << "\n"
          << this->info.string() << "\n\n";

  int total_elements = 0;

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(elastic, acoustic),
      [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        // Getting the number of elements per medium
        int n_elements = this->element_types.get_number_of_elements(medium_tag);
        SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &n_elements, 1,
                                           MPI_INT, MPI_SUM, comm));

        // Printing the number of elements if more than 0
        if (n_elements > 0) {
          // Adding the number of elements to the total
          total_elements += n_elements;

          message << "  Total number of elements of type "
                  << specfem::element::to_string(medium_tag) << " : "
                  << n_elements << "\n";
        };
      });

  if (total_elements == nspec) {
    message << "  All elements accounted for.\n\n";
  } else {
    message << "  NOT ALL ELEMENTS ACCOUNTED FOR:\n";
    message << "   Mesh elements:              " << nspec << "\n";
    message << "   Assembly elements counted:  " << total_elements << "\n";
    message << "   Total unaccounted elements: " << (nspec - total_elements)
            << "\n";
    throw std::runtime_error(message.str());
  }

  return message.str();
}
