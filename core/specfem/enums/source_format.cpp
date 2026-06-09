#include "specfem/enums/source_format.hpp"

std::string
specfem::enums::to_string(const specfem::enums::source_format &fmt) {
  switch (fmt) {
  case specfem::enums::source_format::YAML:
    return "YAML";
  case specfem::enums::source_format::CMTSOLUTION:
    return "CMTSOLUTION";
  case specfem::enums::source_format::FORCESOLUTION:
    return "FORCESOLUTION";
  default:
    return "unknown";
  }
}
