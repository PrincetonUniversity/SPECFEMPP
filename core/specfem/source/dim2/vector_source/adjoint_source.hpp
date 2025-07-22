#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
template <>
class adjoint_source<specfem::dimension::type::dim2>
    : public vector_source<specfem::dimension::type::dim2> {

public:
  adjoint_source() {};

  adjoint_source(
      type_real x, type_real z,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function,
      const std::string &station_name, const std::string &network_name)
      : vector_source(x, z, std::move(forcing_function)),
        station_name(station_name), network_name(network_name) {};

  adjoint_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : station_name(Node["station_name"].as<std::string>()),
        network_name(Node["network_name"].as<std::string>()),
        vector_source(Node, nsteps, dt) {};

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return specfem::wavefield::simulation_field::adjoint;
  }

  std::string print() const override;

  /**
   * @brief Get the force vector
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Force vector
   */
  specfem::kokkos::HostView1d<type_real> get_force_vector() const override;

  /**
   * @brief Get the list of supported media for this source type
   *
   * @return std::vector<specfem::element::medium_tag> list of supported media
   */
  std::vector<specfem::element::medium_tag>
  get_supported_media() const override;

private:
  const static std::string name;
  std::string station_name;
  std::string network_name;
};
} // namespace sources
} // namespace specfem
