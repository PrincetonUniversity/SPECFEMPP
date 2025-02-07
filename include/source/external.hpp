#ifndef _SPECFEM_SOURCES_EXTERNAL_HPP_
#define _SPECFEM_SOURCES_EXTERNAL_HPP_

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/element_types/element_types.hpp"
#include "source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
class external : public source {

public:
  external(){};

  external(YAML::Node &Node, const int nsteps, const type_real dt,
           const specfem::wavefield::simulation_field wavefield_type)
      : wavefield_type(wavefield_type), specfem::sources::source(Node, nsteps,
                                                                 dt){};

  void compute_source_array(
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::element_types &element_types,
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

#endif /* _SPECFEM_SOURCES_EXTERNAL_HPP_ */
