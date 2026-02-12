#pragma once

#include "source.hpp"

#include "specfem_setup.hpp"
#include <yaml-cpp/yaml.h>



template <specfem::element::dimension_tag DimensionTag>
void specfem::sources::source<DimensionTag>::set_source_time_function(
    YAML::Node &Node, const int nsteps, const type_real dt) {

  if (YAML::Node Dirac = Node["Dirac"]) {
    this->source_time_function =
        std::make_unique<specfem::source_time_functions::Dirac>(Dirac, nsteps,
                                                                dt, false);
  } else if (YAML::Node Gaussian = Node["Gaussian"]) {

    // Determine t0_factor based on dimension
    constexpr type_real t0_factor =
        (DimensionTag == specfem::element::dimension_tag::dim2) ? 2.0 : 1.5;

    if (Gaussian["hdur"]) {
      type_real hdur = Gaussian["hdur"].as<type_real>();
      this->source_time_function =
          std::make_unique<specfem::source_time_functions::GaussianHdur>(
              nsteps, dt, hdur,
              Gaussian["tshift"] ? Gaussian["tshift"].as<type_real>() : 0.0,
              Gaussian["factor"].as<type_real>(), false, t0_factor);
      return;
    } else if (Gaussian["f0"]) {
      type_real f0 = Gaussian["f0"].as<type_real>();
      this->source_time_function =
          std::make_unique<specfem::source_time_functions::Gaussian>(
              nsteps, dt, f0,
              Gaussian["tshift"] ? Gaussian["tshift"].as<type_real>() : 0.0,
              Gaussian["factor"].as<type_real>(), false, t0_factor);
      return;
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
  } else if (YAML::Node Heaviside = Node["Heaviside"]) {

    // Determine t0_factor based on dimension
    constexpr type_real t0_factor =
        (DimensionTag == specfem::element::dimension_tag::dim2) ? 2.0 : 1.5;

    this->source_time_function =
        std::make_unique<specfem::source_time_functions::Heaviside>(
            Heaviside, nsteps, dt, false, t0_factor);

  } else if (YAML::Node external = Node["External"]) {
    this->source_time_function =
        std::make_unique<specfem::source_time_functions::external>(external,
                                                                   nsteps, dt);
  } else {
    throw std::runtime_error("Error: source time function not recognized");
  }
}
