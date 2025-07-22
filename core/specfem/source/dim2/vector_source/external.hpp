#pragma once

#include "../vector_source.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
template <>
class external<specfem::dimension::type::dim2>
    : public vector_source<specfem::dimension::type::dim2> {

public:
  external() {};

  external(type_real x, type_real z,
           std::unique_ptr<specfem::forcing_function::stf> forcing_function,
           const specfem::wavefield::simulation_field wavefield_type)
      : vector_source(x, z, std::move(forcing_function)),
        wavefield_type(wavefield_type) {};

  external(YAML::Node &Node, const int nsteps, const type_real dt,
           const specfem::wavefield::simulation_field wavefield_type)
      : wavefield_type(wavefield_type), vector_source(Node, nsteps, dt) {};

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
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
  specfem::wavefield::simulation_field wavefield_type;
  const static std::string name;
  const static std::vector<specfem::element::medium_tag> supported_media;
};
} // namespace sources
} // namespace specfem
