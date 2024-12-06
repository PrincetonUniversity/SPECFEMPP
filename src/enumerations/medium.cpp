#include "enumerations/medium.hpp"

const std::string specfem::element::to_string(
    const specfem::element::medium_tag &medium,
    const specfem::element::property_tag &property_tag) {

  if ((medium == specfem::element::medium_tag::elastic) &&
      (property_tag == specfem::element::property_tag::isotropic)) {
    return "elastic isotropic";
  } else if ((medium == specfem::element::medium_tag::acoustic) &&
             (property_tag == specfem::element::property_tag::isotropic)) {
    return "acoustic isotropic";
  } else {
    return "unknown";
  }
}

const std::string
specfem::element::to_string(const specfem::element::medium_tag &medium) {
  if (medium == specfem::element::medium_tag::elastic) {
    return "elastic";
  } else if (medium == specfem::element::medium_tag::acoustic) {
    return "acoustic";
  } else {
    return "unknown";
  }
}
