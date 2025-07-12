#include "source/force_source.hpp"
#include "enumerations/specfem_enums.hpp"
#include "globals.h"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>

std::string specfem::sources::force::print() const {

  std::ostringstream message;
  message << "- Force Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}

bool specfem::sources::force::operator==(
    const specfem::sources::source &other) const {

  // Try casting the other source to a force source
  const auto *other_source =
      dynamic_cast<const specfem::sources::force *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a force object" << std::endl;
    return false;
  }

  return specfem::utilities::almost_equal(this->x, other_source->x) &&
         specfem::utilities::almost_equal(this->z, other_source->z) &&
         specfem::utilities::almost_equal(this->angle, other_source->angle) &&
         *(this->forcing_function) == *(other_source->forcing_function);
}
bool specfem::sources::force::operator!=(
    const specfem::sources::source &other) const {
  return !(*this == other);
}
