#include "timescheme/timescheme.hpp"
#include "specfem_setup.hpp"
#include <ostream>

std::ostream &
specfem::time_scheme::operator<<(std::ostream &out,
                                 specfem::time_scheme::time_scheme &ts) {
  ts.print(out);

  return out;
}

void specfem::time_scheme::time_scheme::print(std::ostream &out) const {
  out << "Time scheme wasn't initialized properly. Base class being called";

  throw std::runtime_error(
      "Time scheme wasn't initialized properly. Base class being called");
}
