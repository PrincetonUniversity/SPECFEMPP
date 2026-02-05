#include "specfem/element_coupling.hpp"

namespace specfem::element_coupling {
std::string to_string(const interface_tag &interface_tag) {

  std::string interface_string;

  switch (interface_tag) {
  case specfem::element_coupling::interface_tag::acoustic_elastic:
    interface_string = "acoustic_elastic";
    break;
  case specfem::element_coupling::interface_tag::elastic_acoustic:
    interface_string = "elastic_acoustic";
    break;
  default:
    interface_string = "unknown";
    break;
  }

  return interface_string;
}

std::string to_string(const flux_scheme_tag &flux_scheme_tag) {

  std::string flux_scheme_string;

  switch (flux_scheme_tag) {
  case specfem::element_coupling::flux_scheme_tag::natural:
    flux_scheme_string = "natural";
    break;
  case specfem::element_coupling::flux_scheme_tag::symmetric_interior_penalty:
    flux_scheme_string = "symmetric_interior_penalty";
    break;
  default:
    flux_scheme_string = "unknown";
    break;
  }

  return flux_scheme_string;
}

std::ostream &
operator<<(std::ostream &stream,
           const specfem::element_coupling::interface_tag &interface_tag) {
  stream << to_string(interface_tag);
  return stream;
}
std::ostream &
operator<<(std::ostream &stream,
           const specfem::element_coupling::flux_scheme_tag &flux_scheme_tag) {

  stream << to_string(flux_scheme_tag);
  return stream;
}
} // namespace specfem::element_coupling
