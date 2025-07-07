#include "enumerations/wavefield.hpp"

const std::string specfem::wavefield::to_string(
    const specfem::wavefield::type &wavefield_component) {

  std::string component_string;

  switch (wavefield_component) {
  case specfem::wavefield::type::displacement:
    component_string = "displacement";
    break;
  case specfem::wavefield::type::velocity:
    component_string = "velocity";
    break;
  case specfem::wavefield::type::acceleration:
    component_string = "elastic_psv_t";
    break;
  case specfem::wavefield::type::pressure:
    component_string = "pressure";
    break;
  case specfem::wavefield::type::rotation:
    component_string = "rotation";
    break;
  case specfem::wavefield::type::intrinsic_rotation:
    component_string = "intrinsic rotation";
    break;
  case specfem::wavefield::type::curl:
    component_string = "curl";
    break;
  default:
    component_string = "undefined";
    break;
  }

  return component_string;
}
