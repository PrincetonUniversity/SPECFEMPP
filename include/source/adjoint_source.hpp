#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/properties/properties.hpp"
#include "source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
class adjoint_source : public source {

public:
  adjoint_source(){};

  adjoint_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : station_name(Node["station_name"].as<std::string>()),
        network_name(Node["network_name"].as<std::string>()),
        specfem::sources::source(Node, nsteps, dt){};

  void compute_source_array(
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties,
      specfem::kokkos::HostView3d<type_real> source_array) override;

  specfem::wavefield::type get_wavefield_type() const override {
    return specfem::wavefield::type::adjoint;
  }

  std::string print() const override;

private:
  std::string station_name;
  std::string network_name;
};
} // namespace sources
} // namespace specfem
