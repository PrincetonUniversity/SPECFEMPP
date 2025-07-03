#pragma once

#include "source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
class adjoint_source : public source {

public:
  adjoint_source() {};

  adjoint_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : station_name(Node["station_name"].as<std::string>()),
        network_name(Node["network_name"].as<std::string>()),
        specfem::sources::source(Node, nsteps, dt) {};

  void compute_source_array(
      const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
      const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
          &jacobian_matrix,
      const specfem::assembly::element_types &element_types,
      specfem::kokkos::HostView3d<type_real> source_array) override;

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return specfem::wavefield::simulation_field::adjoint;
  }

  std::string print() const override;

private:
  std::string station_name;
  std::string network_name;
};
} // namespace sources
} // namespace specfem
