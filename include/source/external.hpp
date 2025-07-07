#pragma once

#include "source.hpp"
#include "specfem/assembly.hpp"
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

  void compute_source_array(
      const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
      const specfem::assembly::jacobian_matrix &jacobian_matrix,
      const specfem::assembly::element_types &element_types,
      specfem::kokkos::HostView3d<type_real> source_array) override;

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
  }

  std::string print() const override;

private:
  specfem::wavefield::simulation_field wavefield_type;
};
} // namespace sources
} // namespace specfem
