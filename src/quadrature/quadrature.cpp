#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <ostream>

std::ostream &
specfem::quadrature::operator<<(std::ostream &out,
                                specfem::quadrature::quadrature &quad) {
  quad.print(out);

  return out;
}

std::ostream &
specfem::quadrature::operator<<(std::ostream &out,
                                const specfem::quadrature::quadratures &quad) {
  quad.print(out);

  return out;
}

void specfem::quadrature::quadrature::print(std::ostream &out) const {
  out << "Quadrature wasn't initialized properly. Base class being called";

  throw std::runtime_error(
      "Quadrature wasn't initialized properly. Base class being called");
}
