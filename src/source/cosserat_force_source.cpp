#include "source/cosserat_force_source.hpp"
#include "enumerations/specfem_enums.hpp"
#include "globals.h"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include <cmath>

std::string specfem::sources::cosserat_force::print() const {

  std::ostringstream message;
  message << "- Cosserat Force Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Source Angle: " << type_real(this->angle) << "\n"
          << "    Source f: " << type_real(this->f) << "\n"
          << "    Source fc: " << type_real(this->fc) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}

bool specfem::sources::cosserat_force::operator==(
    const specfem::sources::source &other) const {

  // Try casting the other source to a cosserat_force source
  const auto *other_source =
      dynamic_cast<const specfem::sources::cosserat_force *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a cosserat_force object" << std::endl;
    return false;
  }

  bool internal =
      specfem::utilities::almost_equal(this->f, other_source->f) &&
      specfem::utilities::almost_equal(this->x, other_source->x) &&
      specfem::utilities::almost_equal(this->z, other_source->z) &&
      specfem::utilities::almost_equal(this->angle, other_source->angle);

  if (!internal) {
    std::cout << "Cosserat force sources not equal" << std::endl;
  }

  return internal &&
         (*(this->forcing_function) == *(other_source->forcing_function));
}
bool specfem::sources::cosserat_force::operator!=(
    const specfem::sources::source &other) const {
  return !(*this == other);
}
