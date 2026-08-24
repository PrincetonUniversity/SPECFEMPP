#include "specfem/enums.hpp"

const std::string specfem::enums::to_string(
    const specfem::enums::wavefield &wavefield_component) {

  std::string component_string;

  switch (wavefield_component) {
  case specfem::enums::wavefield::displacement:
    component_string = "displacement";
    break;
  case specfem::enums::wavefield::velocity:
    component_string = "velocity";
    break;
  case specfem::enums::wavefield::acceleration:
    component_string = "acceleration";
    break;
  case specfem::enums::wavefield::pressure:
    component_string = "pressure";
    break;
  case specfem::enums::wavefield::rotation:
    component_string = "rotation";
    break;
  case specfem::enums::wavefield::intrinsic_rotation:
    component_string = "intrinsic rotation";
    break;
  case specfem::enums::wavefield::curl:
    component_string = "curl";
    break;
  default:
    component_string = "undefined";
    break;
  }

  return component_string;
}
