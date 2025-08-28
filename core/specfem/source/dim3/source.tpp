
#include "specfem/source.hpp"
#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <yaml-cpp/yaml.h>
#include <cmath>


template <specfem::dimension::type DimensionTag>
template <specfem::dimension::type U, typename std::enable_if<U == specfem::dimension::type::dim3>::type*>
specfem::sources::source<DimensionTag>::source(
    YAML::Node &Node, const int nsteps, const type_real dt)
    : global_coordinates(Node["x"].as<type_real>(), Node["y"].as<type_real>(),
                         Node["z"].as<type_real>()) {

  // Read source time function
  if (YAML::Node Dirac = Node["Dirac"]) {
    this->forcing_function = std::make_unique<specfem::forcing_function::Dirac>(
        Dirac, nsteps, dt, false);
  } else if (YAML::Node Ricker = Node["Ricker"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::Ricker>(Ricker, nsteps, dt,
                                                            false);
  } else if (YAML::Node dGaussian = Node["dGaussian"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::dGaussian>(
            dGaussian, nsteps, dt, false);
  } else if (YAML::Node external = Node["External"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::external>(external, nsteps,
                                                              dt);
  } else {
    throw std::runtime_error("Error: source time function not recognized");
  }

  return;
}
