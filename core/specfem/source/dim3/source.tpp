
#include "specfem/source.hpp"
#include "kokkos_abstractions.h"

#include "specfem/setup.hpp"
#include <yaml-cpp/yaml.h>
#include <cmath>


template <specfem::element::dimension_tag DimensionTag>
template <specfem::element::dimension_tag U, typename std::enable_if<U == specfem::element::dimension_tag::dim3>::type*>
specfem::sources::source<DimensionTag>::source(
    YAML::Node &Node, const int nsteps, const type_real dt)
    : global_coordinates(Node["x"].as<type_real>(), Node["y"].as<type_real>(),
                         Node["z"].as<type_real>()) {

  // Read source time function
  if (YAML::Node Dirac = Node["Dirac"]) {
    this->source_time_function = std::make_unique<specfem::source_time_functions::Dirac>(
        Dirac, nsteps, dt, false);
  } else if (YAML::Node Ricker = Node["Ricker"]) {
    this->source_time_function =
        std::make_unique<specfem::source_time_functions::Ricker>(Ricker, nsteps, dt,
                                                            false);
  } else if (YAML::Node dGaussian = Node["dGaussian"]) {
    this->source_time_function =
        std::make_unique<specfem::source_time_functions::dGaussian>(
            dGaussian, nsteps, dt, false);
  } else if (YAML::Node external = Node["External"]) {
    this->source_time_function =
        std::make_unique<specfem::source_time_functions::external>(external, nsteps,
                                                              dt);
  } else {
    throw std::runtime_error("Error: source time function not recognized");
  }

  return;
}
