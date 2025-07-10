#include "source/adjoint_source.hpp"
#include "algorithms/locate_point.hpp"
#include "globals.h"

std::string specfem::sources::adjoint_source::print() const {

  std::ostringstream message;
  message << "- Adjoint Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";
  return message.str();
}
