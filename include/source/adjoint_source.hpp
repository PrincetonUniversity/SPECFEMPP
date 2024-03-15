#ifndef _SOURCES_ADJOINT_SOURCE_HPP_
#define _SOURCES_ADJOINT_SOURCE_HPP_

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/properties/properties.hpp"
#include "source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
class adjoint_source : public source {
public:
  adjoint_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : specfem::sources::source(Node, nsteps, dt){};

  void compute_source_array(
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties,
      specfem::kokkos::HostView3d<type_real> source_array) override;
};
} // namespace sources
} // namespace specfem

#endif /* _SOURCES_ADJOINT_SOURCE_HPP_ */
