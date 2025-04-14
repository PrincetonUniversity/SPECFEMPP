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
  this->properties = { this->mesh.nspec,          this->mesh.ngllz,
                       this->mesh.ngllx,          this->element_types,
                       this->mesh.mapping,        mesh.materials,
                       property_reader != nullptr };
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

  // Checks
  if (nelastic_sh > 0 && nacoustic > 0) {
    std::ostringstream msg;
    msg << "Elastic SH and acoustic elements cannot be mixed in the same "
        << "domain. We currently do not support SH and pressure wave coupling. "
        << "Please check your MESHFEM input file.";

    throw std::runtime_error(msg.str());
  }

  const auto pe_stacey_elements = this->element_types.get_elements_on_device(
      specfem::element::medium_tag::poroelastic,
      specfem::element::property_tag::isotropic,
      specfem::element::boundary_tag::stacey);

  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::poroelastic,
                             specfem::element::property_tag::isotropic, false>
      point_values;

  specfem::compute::max(pe_stacey_elements, this->properties, point_values);

  if ((pe_stacey_elements.extent(0) > 0) &&
      std::abs(point_values.eta_f()) > 1e-6) {
    std::ostringstream msg;
    msg << "Warning: The poroelastic model with Stacey BCs can be numerically "
           "error prone. Please make sure there are no spurious reflections "
           "off the boundary";

    std::cerr << msg.str();
  }

  return;
}


std::string specfem::compute::assembly::print() const {
  std::ostringstream message;
  message << "Assembly information:\n"
          << "------------------------------\n"
          << "Total number of spectral elements : " << this->mesh.nspec << "\n"
          << "Total number of geometric points : " << this->mesh.ngllz << "\n";

  int total_elements = 0;
  
  FOR_EACH_MATERIAL_SYSTEM(
    IN((DIMENSION_TAG_DIM2),
        (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC,
        MEDIUM_TAG_POROELASTIC),
        (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
    DECLARE((int, n_elements)))
  
  FOR_EACH_MATERIAL_SYSTEM(
    IN((DIMENSION_TAG_DIM2),
        (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC,
        MEDIUM_TAG_POROELASTIC),
        (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
    CAPTURE(n_elements)
    {
      // Getting the number of elements per medium
      _n_elements_ = this->element_types.get_number_of_elements(
          _medium_tag_, _property_tag_);

      // Adding the number of elements to the total
      total_elements += _n_elements_;
      
      // Printing the number of elements if more than 0
      if (_n_elements_ > 0) {
        message << "   Total number of elements of type "
                << specfem::element::to_string(_medium_tag_, _property_tag_)
                << " : " << _n_elements_ << "\n";
      };

    })
  
  bool is_sh = false;
  bool is_psv = false;

  FOR_EACH_MATERIAL_SYSTEM(
    IN((DIMENSION_TAG_DIM2),
        (MEDIUM_TAG_ELASTIC_PSV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC,
        MEDIUM_TAG_POROELASTIC),
        (PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC)),
    CAPTURE(n_elements) 
    {
      if (_medium_tag_ == specfem::element::medium_tag::elastic_sh) {
        if (_n_elements_ > 0) {
          is_sh = true;
        }
      } else if (_medium_tag_ == specfem::element::medium_tag::elastic_psv) {
        if (_n_elements_ > 0) {
          is_psv = true;
        }
      } 
    })
  
  if (is_sh && is_psv) {
    message << "   WARNING: This should not appear something's off in the code's handling of polarization.\n";
  } else if (is_sh) {
    message << "   Elastic media will simulate SH polarized waves\n";
  } else if (is_psv) {
    message << "   Elastic media will simulate P-SV polarized waves\n";
  }

  return message.str();
}
