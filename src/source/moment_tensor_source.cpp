#include "source/moment_tensor_source.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
// #include "utilities.cpp"
#include "yaml-cpp/yaml.h"
#include <cmath>

std::string specfem::sources::moment_tensor::print() const {
  std::ostringstream message;
  message << "- Moment Tensor Source: \n"
          << "    Source Location: \n"
          << "      x = " << this->x << "\n"
          << "      z = " << this->z << "\n"
          << "    Moment Tensor: \n"
          << "      Mxx, Mzz, Mxz = " << this->Mxx << ", " << this->Mzz << ", "
          << this->Mxz << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}

bool specfem::sources::moment_tensor::operator==(
    const specfem::sources::source &other) const {

  // Try casting the other source to a moment tensor source
  const auto *other_source =
      dynamic_cast<const specfem::sources::moment_tensor *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a moment tensor object" << std::endl;
    return false;
  }

  bool internal =
      specfem::utilities::almost_equal(this->Mxx, other_source->Mxx) &&
      specfem::utilities::almost_equal(this->Mxz, other_source->Mxz) &&
      specfem::utilities::almost_equal(this->Mzz, other_source->Mzz) &&
      specfem::utilities::almost_equal(this->x, other_source->x) &&
      specfem::utilities::almost_equal(this->z, other_source->z);

  if (!internal) {
    std::cout << "Moment tensor source not equal" << std::endl;
  }

  return internal &&
         (*(this->forcing_function) == *(other_source->forcing_function));
}
bool specfem::sources::moment_tensor::operator!=(
    const specfem::sources::source &other) const {
  return !(*this == other);
}
