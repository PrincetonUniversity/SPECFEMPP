#include "enumerations/specfem_enums.hpp"
#include "source/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include <cmath>

std::string specfem::sources::external::print() const {

  std::ostringstream message;
  message << "- External Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}
