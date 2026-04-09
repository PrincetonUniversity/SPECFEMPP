#include "to_string.hpp"

const std::string specfem::element_connections::to_string(
    const specfem::element_connections::type &conn) {
  switch (conn) {
  case specfem::element_connections::type::strongly_conforming:
    return "strongly_conforming";
  case specfem::element_connections::type::weakly_conforming:
    return "weakly_conforming";
  case specfem::element_connections::type::nonconforming:
    return "nonconforming";
  default:
    throw std::runtime_error(
        std::string(
            "specfem::element_connections::to_string does not handle ") +
        std::to_string(static_cast<int>(conn)));
    return "!ERR";
  }
}
