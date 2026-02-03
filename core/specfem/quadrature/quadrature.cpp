#include "quadrature.hpp"
#include "specfem_setup.hpp"
#include <ostream>

std::ostream &
specfem::quadrature::operator<<(std::ostream &out,
                                specfem::quadrature::quadrature &quad) {
  quad.print(out);

  return out;
}

void specfem::quadrature::quadrature::print(std::ostream &out) const {
  auto outstring = this->to_string();
  throw std::runtime_error(outstring);
}

std::string specfem::quadrature::quadrature::to_string() const {
  return "Quadrature wasn't initialized properly. Base class being called";
}
