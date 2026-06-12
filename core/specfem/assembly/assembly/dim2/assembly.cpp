#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/info.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/tag_dispatch.hpp"

specfem::assembly::assembly<specfem::element::dimension_tag::dim2>::assembly(
    const specfem::mesh::mesh<dimension_tag> &mesh,
    const specfem::quadrature::quadratures &quadratures,
    std::vector<std::shared_ptr<specfem::sources::source<dimension_tag>>>
        &sources,
    const std::vector<std::shared_ptr<
        specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>>
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
  this->mesh = { mesh.tags, mesh.control_nodes, quadratures,
                 mesh.adjacency_graph };
  this->element_types = { this->mesh.nspec, this->mesh.element_grid, this->mesh,
                          mesh.tags };
  this->element_intersections = { this->mesh.element_grid.ngllx,
                                  this->mesh.element_grid.ngllz, this->mesh,
                                  this->element_types, flux_scheme_config };
  this->jacobian_matrix = { this->mesh };

  this->field_derivative_storage = { this->element_types, this->mesh.nspec,
                                     this->mesh.element_grid.ngllz,
                                     this->mesh.element_grid.ngllx };

  this->kernels = { this->element_types };
  this->sources = {
    sources, this->mesh, this->jacobian_matrix, this->element_types,
    t0,      dt,         max_timesteps
  };
  this->receivers = { this->mesh.nspec,
                      this->mesh.element_grid.ngllz,
                      this->mesh.element_grid.ngllz,
                      max_sig_step,
                      dt,
                      t0,
                      nsteps_between_samples,
                      receivers,
                      stypes,
                      this->mesh,
                      mesh.tags,
                      this->element_types };
  this->boundaries = { this->mesh.nspec,
                       this->mesh.element_grid.ngllz,
                       this->mesh.element_grid.ngllx,
                       mesh,
                       this->mesh,
                       this->jacobian_matrix };
  this->conforming_interfaces = { this->mesh.element_grid.ngllz,
                                  this->mesh.element_grid.ngllx,
                                  this->element_intersections,
                                  this->jacobian_matrix, this->mesh };
  this->nonconforming_interfaces = { this->mesh.element_grid.ngllz,
                                     this->mesh.element_grid.ngllx,
                                     this->element_intersections, this->mesh,
                                     flux_scheme_config };
  this->fields = { this->mesh, this->element_types, simulation };

  this->attenuation = { mesh.attenuation, dt, this->mesh, this->element_types,
                        mesh.materials };

  this->properties = { this->element_types, this->mesh, mesh.materials,
                       property_reader != nullptr };

  this->info = { this->mesh, this->properties, this->element_types };

  if (allocate_boundary_values)
    this->boundary_values = { max_timesteps, this->mesh, this->element_types,
                              this->boundaries };

  /// Add some domain checks here for SH domains
  const int nelastic_sh = this->element_types.get_number_of_elements(
      specfem::element::medium_tag::elastic_sh);

  const int nacoustic = this->element_types.get_number_of_elements(
      specfem::element::medium_tag::acoustic);

  // Checks
  if (nelastic_sh > 0 && nacoustic > 0) {
    std::ostringstream msg;
    msg << "Elastic SH and acoustic elements cannot be mixed in the same "
        << "domain. We currently do not support SH and pressure wave coupling. "
        << "Please check your MESHFEM input file.";

    throw std::runtime_error(msg.str());
  }

  this->check_jacobian_matrix();

  return;
}

std::string
specfem::assembly::assembly<specfem::element::dimension_tag::dim2>::print()
    const {
  std::ostringstream message;
  int nspec = this->mesh.nspec;
  int nglob = this->mesh.nglob;

  const auto comm = specfem::MPI::communicator();
  SPECFEM_MPI_SAFECALL(
      MPI_Allreduce(MPI_IN_PLACE, &nspec, 1, MPI_INT, MPI_SUM, comm));
  SPECFEM_MPI_SAFECALL(
      MPI_Allreduce(MPI_IN_PLACE, &nglob, 1, MPI_INT, MPI_SUM, comm));

  message << "Assembly information:\n"
          << "------------------------------\n"
          << "Total number of spectral elements : " << nspec << "\n"
          << "Total number of geometric points : "
          << this->mesh.element_grid.ngllz << "\n"
          << "Total number of distinct quadrature points : " << nglob << "\n"
          << this->info.string() << "\n";

  int total_elements = 0;

  bool is_sh = false;
  bool is_psv = false;

  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim2) * MEDIUM_SET(elastic_psv, elastic_sh, acoustic,
                                       poroelastic, elastic_psv_t),
      [&]<typename ElementTags>() {
        // Getting the number of elements per medium
        int n_elements =
            this->element_types.get_number_of_elements(ElementTags::medium_tag);
        SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &n_elements, 1,
                                           MPI_INT, MPI_SUM, comm));

        // Printing the number of elements if more than 0
        if (n_elements > 0) {
          // Adding the number of elements to the total
          total_elements += n_elements;

          message << "   Total number of elements of type "
                  << specfem::element::to_string(ElementTags::medium_tag)
                  << " : " << n_elements << "\n";
          if (ElementTags::medium_tag ==
              specfem::element::medium_tag::elastic_sh) {
            is_sh = true;
          } else if (ElementTags::medium_tag ==
                     specfem::element::medium_tag::elastic_psv) {
            is_psv = true;
          }
        };
      });

  if (is_sh && is_psv) {
    message << "   WARNING: This should not appear something's off in the "
               "code's handling of polarization.\n";
  } else if (is_sh) {
    message << "   Elastic media will simulate SH polarized waves\n";
  } else if (is_psv) {
    message << "   Elastic media will simulate P-SV polarized waves\n";
  }

  if (total_elements == nspec) {
    message << "  All elements accounted for.\n";
  } else {
    message << " NOT ALL ELEMENTS ACCOUNTED FOR\n";
    message << "  Mesh elements:              " << nspec << "\n";
    message << "  Assembly elements counted:  " << total_elements << "\n";
    message << "  Total unaccounted elements: " << (nspec - total_elements)
            << "\n";
    throw std::runtime_error(message.str());
  }

  return message.str();
}
