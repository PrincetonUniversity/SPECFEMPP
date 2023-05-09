#include "specfem_setup.hpp"
#include "timescheme/interface.hpp"
#include <ostream>

std::ostream &
specfem::TimeScheme::operator<<(std::ostream &out,
                                specfem::TimeScheme::TimeScheme &ts) {
  ts.print(out);

  return out;
}

void specfem::TimeScheme::TimeScheme::print(std::ostream &out) const {
  out << "Time scheme wasn't initialized properly. Base class being called";

  throw std::runtime_error(
      "Time scheme wasn't initialized properly. Base class being called");
}
