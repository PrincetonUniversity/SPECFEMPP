#include "algorithms/locate_point.hpp"
#include "enumerations/specfem_enums.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
// #include "utilities.cpp"
#include "yaml-cpp/yaml.h"
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
