
#include "specfem/source.hpp"

#include "specfem/setup.hpp"
#include <cmath>
#include <yaml-cpp/yaml.h>

template <specfem::element::dimension_tag DimensionTag>
template <specfem::element::dimension_tag U, typename std::enable_if<U == specfem::element::dimension_tag::dim3>::type*>
specfem::sources::source<DimensionTag>::source(
    YAML::Node &Node, const int nsteps, const type_real dt)
    : global_coordinates(Node["x"].as<type_real>(), Node["y"].as<type_real>(),
                         Node["z"].as<type_real>()) {

  // Read source time function
  if (YAML::Node Dirac = Node["Dirac"]) {
    this->source_time_function =
        std::make_unique<specfem::source_time_functions::Dirac>(Dirac, nsteps,
                                                                dt, false);
  } else if (YAML::Node Gaussian = Node["Gaussian"]) {
    constexpr type_real t0_factor = 1.5;
    if (Gaussian["hdur"]) {
      this->source_time_function =
          std::make_unique<specfem::source_time_functions::GaussianHdur>(
              nsteps, dt, Gaussian["hdur"].as<type_real>(),
              Gaussian["tshift"] ? Gaussian["tshift"].as<type_real>() : 0.0,
              Gaussian["factor"].as<type_real>(), false, t0_factor);
    } else if (Gaussian["f0"]) {
      this->source_time_function =
          std::make_unique<specfem::source_time_functions::Gaussian>(
              nsteps, dt, Gaussian["f0"].as<type_real>(),
              Gaussian["tshift"] ? Gaussian["tshift"].as<type_real>() : 0.0,
              Gaussian["factor"].as<type_real>(), false, t0_factor);
    } else {
      throw std::runtime_error(
          "Error: Gaussian source time function requires either 'hdur' or 'f0' "
          "to be specified.");
    }
  } else if (YAML::Node Ricker = Node["Ricker"]) {
    this->source_time_function =
        std::make_unique<specfem::source_time_functions::Ricker>(Ricker, nsteps,
                                                                 dt, false);
  } else if (YAML::Node dGaussian = Node["dGaussian"]) {
    this->source_time_function =
        std::make_unique<specfem::source_time_functions::dGaussian>(
            dGaussian, nsteps, dt, false);
  } else if (YAML::Node external = Node["External"]) {
    this->source_time_function =
        std::make_unique<specfem::source_time_functions::external>(external,
                                                                   nsteps, dt);
  } else {
    throw std::runtime_error("Error: source time function not recognized");
  }

  return;
}
