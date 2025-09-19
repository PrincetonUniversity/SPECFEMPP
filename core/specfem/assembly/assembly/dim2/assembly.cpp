#include "assembly.hpp"
#include "enumerations/interface.hpp"
#include "io/reader.hpp"
#include "mesh/mesh.hpp"

specfem::assembly::assembly<specfem::dimension::type::dim2>::assembly(
    const specfem::mesh::mesh<dimension_tag> &mesh,
    const specfem::quadrature::quadratures &quadratures,
    std::vector<std::shared_ptr<specfem::sources::source<dimension_tag> > >
        &sources,
    const std::vector<std::shared_ptr<
        specfem::receivers::receiver<specfem::dimension::type::dim2> > >
        &receivers,
    const std::vector<specfem::wavefield::type> &stypes, const type_real t0,
    const type_real dt, const int max_timesteps, const int max_sig_step,
    const int nsteps_between_samples,
    const specfem::simulation::type simulation,
    const bool allocate_boundary_values,
    const std::shared_ptr<specfem::io::reader> &property_reader) {
  this->mesh = { mesh.tags, mesh.control_nodes, quadratures,
                 mesh.adjacency_graph };
  this->element_types = { this->mesh.nspec, this->mesh.element_grid.ngllz,
                          this->mesh.element_grid.ngllx, this->mesh,
                          mesh.tags };
  this->edge_types = { this->mesh.element_grid.ngllx,
                       this->mesh.element_grid.ngllz, this->mesh,
                       this->element_types, mesh.coupled_interfaces };
  this->jacobian_matrix = { this->mesh };
  this->properties = { this->mesh.nspec,
                       this->mesh.element_grid.ngllz,
                       this->mesh.element_grid.ngllx,
                       this->element_types,
                       this->mesh,
                       mesh.materials,
                       property_reader != nullptr };
  this->kernels = { this->mesh.nspec, this->mesh.element_grid.ngllz,
                    this->mesh.element_grid.ngllx, this->element_types };
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
  this->coupled_interfaces = { this->mesh.element_grid.ngllz,
                               this->mesh.element_grid.ngllx, this->edge_types,
                               this->jacobian_matrix, this->mesh };
  this->fields = { this->mesh, this->element_types, simulation };

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
specfem::assembly::assembly<specfem::dimension::type::dim2>::print() const {
  std::ostringstream message;
  message << "Assembly information:\n"
          << "------------------------------\n"
          << "Total number of spectral elements : " << this->mesh.nspec << "\n"
          << "Total number of geometric points : "
          << this->mesh.element_grid.ngllz << "\n"
          << "Total number of distinct quadrature points : " << this->mesh.nglob
          << "\n";

  int total_elements = 0;

  bool is_sh = false;
  bool is_psv = false;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      {
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
          if (_medium_tag_ == specfem::element::medium_tag::elastic_sh) {
            is_sh = true;
          } else if (_medium_tag_ ==
                     specfem::element::medium_tag::elastic_psv) {
            is_psv = true;
          }
        };
      })

  if (is_sh && is_psv) {
    message << "   WARNING: This should not appear something's off in the "
               "code's handling of polarization.\n";
  } else if (is_sh) {
    message << "   Elastic media will simulate SH polarized waves\n";
  } else if (is_psv) {
    message << "   Elastic media will simulate P-SV polarized waves\n";
  }

  return message.str();
}
