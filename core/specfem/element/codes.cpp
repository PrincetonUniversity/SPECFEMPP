#include "codes.hpp"
#include "to_string.hpp"
#include <stdexcept>
#include <string>

specfem::element::region_tag
specfem::element::region_tag_from_code(const int code) {
  switch (code) {
  case 1:
    return specfem::element::region_tag::crust_mantle;
  case 2:
    return specfem::element::region_tag::outer_core;
  case 3:
    return specfem::element::region_tag::inner_core;
  default:
    throw std::runtime_error("Unknown region code " + std::to_string(code));
  }
}

specfem::element::medium_tag
specfem::element::medium_tag_from_code(const int code) {
  switch (code) {
  case 1:
    return specfem::element::medium_tag::acoustic;
  case 2:
    return specfem::element::medium_tag::elastic;
  default:
    throw std::runtime_error("Unknown medium code " + std::to_string(code));
  }
}

specfem::element::property_tag
specfem::element::property_tag_from_code(const int code) {
  switch (code) {
  case 0:
    return specfem::element::property_tag::isotropic;
  case 1:
    return specfem::element::property_tag::anisotropic;
  default:
    throw std::runtime_error("Unknown property code " + std::to_string(code));
  }
}

int specfem::element::to_code(const specfem::element::region_tag &region) {
  switch (region) {
  case specfem::element::region_tag::crust_mantle:
    return 1;
  case specfem::element::region_tag::outer_core:
    return 2;
  case specfem::element::region_tag::inner_core:
    return 3;
  default:
    throw std::runtime_error("Region tag without database code: " +
                             specfem::element::to_string(region));
  }
}

int specfem::element::to_code(const specfem::element::medium_tag &medium) {
  switch (medium) {
  case specfem::element::medium_tag::acoustic:
    return 1;
  case specfem::element::medium_tag::elastic:
    return 2;
  default:
    throw std::runtime_error("Medium tag without database code: " +
                             specfem::element::to_string(medium));
  }
}

int specfem::element::to_code(const specfem::element::property_tag &property) {
  switch (property) {
  case specfem::element::property_tag::isotropic:
    return 0;
  case specfem::element::property_tag::anisotropic:
    return 1;
  default:
    throw std::runtime_error("Property tag without database code: " +
                             specfem::element::to_string(property));
  }
}
