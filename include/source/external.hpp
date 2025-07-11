#pragma once

#include "source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
class external : public source {

public:
  external() {};

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

private:
  specfem::wavefield::simulation_field wavefield_type;
};
} // namespace sources
} // namespace specfem
