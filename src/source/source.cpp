#include "kokkos_abstractions.h"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <cmath>

specfem::sources::source::source(YAML::Node &Node, const type_real dt)
    : x(Node["x"].as<type_real>()), z(Node["z"].as<type_real>()) {

  // Read source time function
  if (YAML::Node Dirac = Node["Dirac"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::Dirac>(Dirac, dt, false);
  } else if (YAML::Node Ricker = Node["Ricker"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::Ricker>(Ricker, dt, false);
  } else {
    throw std::runtime_error("Only Dirac and Ricker sources are supported.");
  }

  return;
}
