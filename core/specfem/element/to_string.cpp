#include "to_string.hpp"
#include <stdexcept>
#include <string>

const std::string specfem::element::to_string(
    const specfem::element::medium_tag &medium,
    const specfem::element::property_tag &property_tag,
    const specfem::element::attenuation_tag &attenuation_tag) {

  std::string medium_string = specfem::element::to_string(medium);
  std::string property_string = specfem::element::to_string(property_tag);
  std::string attenuation_string = specfem::element::to_string(attenuation_tag);

  return medium_string + "_" + property_string + "_" + attenuation_string;
}

const std::string
specfem::element::to_string(const specfem::element::medium_tag &medium,
                            const specfem::element::property_tag &property,
                            const specfem::element::boundary_tag &boundary) {
  std::string medium_string = specfem::element::to_string(medium);
  std::string property_string = specfem::element::to_string(property);
  std::string boundary_string = specfem::element::to_string(boundary);

  return medium_string + "_" + property_string + "_" + boundary_string;
}

const std::string
specfem::element::to_string(const specfem::element::medium_tag &medium) {

  std::string medium_string;

  switch (medium) {
  case specfem::element::medium_tag::elastic:
    medium_string = "elastic";
    break;
  case specfem::element::medium_tag::elastic_psv:
    medium_string = "elastic_psv";
    break;
  case specfem::element::medium_tag::elastic_sh:
    medium_string = "elastic_sh";
    break;
  case specfem::element::medium_tag::elastic_psv_t:
    medium_string = "elastic_psv_t";
    break;
  case specfem::element::medium_tag::acoustic:
    medium_string = "acoustic";
    break;
  case specfem::element::medium_tag::electromagnetic_te:
    medium_string = "electromagnetic_te";
    break;
  case specfem::element::medium_tag::poroelastic:
    medium_string = "poroelastic";
    break;
  default:
    medium_string = "unknown";
    break;
  }

  return medium_string;
}

const std::string
specfem::element::to_string(const specfem::element::property_tag &property) {

  std::string property_string;

  switch (property) {
  case specfem::element::property_tag::isotropic:
    property_string = "isotropic";
    break;
  case specfem::element::property_tag::anisotropic:
    property_string = "anisotropic";
    break;
  default:
    property_string = "unknown";
    break;
  }

  return property_string;
}

const std::string
specfem::element::to_string(const specfem::element::boundary_tag &boundary) {

  std::string boundary_string;

  switch (boundary) {
  case specfem::element::boundary_tag::none:
    boundary_string = "none";
    break;
  case specfem::element::boundary_tag::acoustic_free_surface:
    boundary_string = "acoustic_free_surface";
    break;
  case specfem::element::boundary_tag::stacey:
    boundary_string = "stacey";
    break;
  case specfem::element::boundary_tag::composite_stacey_dirichlet:
    boundary_string = "composite_stacey_dirichlet";
    break;
  default:
    boundary_string = "unknown";
    break;
  }

  return boundary_string;
}

const std::string specfem::element::to_string(
    const specfem::element::attenuation_tag &attenuation) {

  std::string attenuation_string;

  switch (attenuation) {
  case specfem::element::attenuation_tag::none:
    attenuation_string = "none";
    break;
  case specfem::element::attenuation_tag::constant_isotropic:
    attenuation_string = "constant_isotropic";
    break;
  default:
    attenuation_string = "unknown";
    break;
  }

  return attenuation_string;
}

specfem::element::medium_tag
specfem::element::from_string(const std::string &medium_tag) {
  if (medium_tag == "elastic_psv") {
    return specfem::element::medium_tag::elastic_psv;
  } else if (medium_tag == "elastic_sh") {
    return specfem::element::medium_tag::elastic_sh;
  } else if (medium_tag == "elastic_psv_t") {
    return specfem::element::medium_tag::elastic_psv_t;
  } else if (medium_tag == "acoustic") {
    return specfem::element::medium_tag::acoustic;
  } else if (medium_tag == "poroelastic") {
    return specfem::element::medium_tag::poroelastic;
  } else if (medium_tag == "electromagnetic_te") {
    return specfem::element::medium_tag::electromagnetic_te;
  } else if (medium_tag == "elastic") {
    return specfem::element::medium_tag::elastic;
  } else if (medium_tag == "elastic_spin") {
    return specfem::element::medium_tag::elastic_spin;
  } else if (medium_tag == "electromagnetic") {
    return specfem::element::medium_tag::electromagnetic;
  } else {
    // If the medium tag is not recognized, throw an error
    // This is to ensure that the function always returns a valid medium tag
    // or throws an error if the input is invalid.
    throw std::runtime_error("Unknown medium tag: " + medium_tag);
  }
}
