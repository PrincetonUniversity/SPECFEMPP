#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
class adjoint_source : public source {

public:
  adjoint_source() {};

  adjoint_source(
      type_real x, type_real z,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function,
      const std::string &station_name, const std::string &network_name)
      : source(x, z, std::move(forcing_function)), station_name(station_name),
        network_name(network_name) {};

  adjoint_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : station_name(Node["station_name"].as<std::string>()),
        network_name(Node["network_name"].as<std::string>()),
        specfem::sources::source(Node, nsteps, dt) {};

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return specfem::wavefield::simulation_field::adjoint;
  }

  std::string print() const override;

  /**
   * @brief Get the source type
   *
   * @return source_type type of source
   */
  source_type get_source_type() const override {
    return source_type::adjoint_source;
  }

  /**
   * @brief Get the force vector
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Force vector
   */
  specfem::kokkos::HostView1d<type_real> get_force_vector() const;

private:
  std::string station_name;
  std::string network_name;
};
} // namespace sources
} // namespace specfem
