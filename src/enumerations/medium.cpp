#include "enumerations/medium.hpp"

const std::string
specfem::element::to_string(const specfem::element::medium_tag &medium,
                            const specfem::element::property_tag &property,
                            const specfem::element::boundary_tag &boundary) {
  std::string medium_string;
  std::string property_string;
  std::string boundary_string;

  switch (medium) {
  case specfem::element::medium_tag::elastic_sv:
    medium_string = "elastic_sv";
    break;
  case specfem::element::medium_tag::elastic_sh:
    medium_string = "elastic_sh";
    break;
  case specfem::element::medium_tag::acoustic:
    medium_string = "acoustic";
    break;
  case specfem::element::medium_tag::electromagnetic_sv:
    medium_string = "electromagnetic_sv";
    break;
  default:
    medium_string = "unknown";
    break;
  }

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

  return medium_string + "_" + property_string + "_" + boundary_string;
}

const std::string specfem::element::to_string(
    const specfem::element::medium_tag &medium,
    const specfem::element::property_tag &property_tag) {

  std::string medium_string;
  std::string property_string;

  switch (medium) {
  case specfem::element::medium_tag::elastic_sv:
    medium_string = "elastic_sv";
    break;
  case specfem::element::medium_tag::elastic_sh:
    medium_string = "elastic_sh";
    break;
  case specfem::element::medium_tag::acoustic:
    medium_string = "acoustic";
    break;
  case specfem::element::medium_tag::electromagnetic_sv:
    medium_string = "electromagnetic_sv";
    break;
  default:
    medium_string = "unknown";
    break;
  }

  switch (property_tag) {
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

  return medium_string + "_" + property_string;
}

const std::string
specfem::element::to_string(const specfem::element::medium_tag &medium) {

  std::string medium_string;

  switch (medium) {
  case specfem::element::medium_tag::elastic_sv:
    medium_string = "elastic_sv";
    break;
  case specfem::element::medium_tag::elastic_sh:
    medium_string = "elastic_sh";
    break;
  case specfem::element::medium_tag::acoustic:
    medium_string = "acoustic";
    break;
  case specfem::element::medium_tag::electromagnetic_sv:
    medium_string = "electromagnetic_sv";
    break;
  default:
    medium_string = "unknown";
    break;
  }

  return medium_string;
}
