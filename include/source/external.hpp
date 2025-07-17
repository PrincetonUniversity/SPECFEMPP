#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
class external : public source {

public:
  external() {};

  external(type_real x, type_real z,
           std::unique_ptr<specfem::forcing_function::stf> forcing_function,
           const specfem::wavefield::simulation_field wavefield_type)
      : source(x, z, std::move(forcing_function)),
        wavefield_type(wavefield_type) {};

  external(YAML::Node &Node, const int nsteps, const type_real dt,
           const specfem::wavefield::simulation_field wavefield_type)
      : wavefield_type(wavefield_type),
        specfem::sources::source(Node, nsteps, dt) {};

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
  }

  std::string print() const override;

  /**
   * @brief Get the source type
   *
   * @return source_type type of source
   */
  source_type get_source_type() const override {
    return source_type::external_source;
  }

  /**
   * @brief Get the force vector
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Force vector
   */
  specfem::kokkos::HostView1d<type_real> get_force_vector() const;

private:
  specfem::wavefield::simulation_field wavefield_type;
};
} // namespace sources
} // namespace specfem
