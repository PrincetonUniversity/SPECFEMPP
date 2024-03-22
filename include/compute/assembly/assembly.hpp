#ifndef _COMPUTE_ASSEMBLY_HPP
#define _COMPUTE_ASSEMBLY_HPP

#include "compute/boundaries/boundaries.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/compute_receivers.hpp"
#include "compute/compute_sources.hpp"
#include "compute/coupled_interfaces/coupled_interfaces.hpp"
#include "compute/fields/fields.hpp"
#include "compute/properties/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "mesh/mesh.hpp"
#include "receiver/interface.hpp"
#include "source/interface.hpp"

namespace specfem {
namespace compute {
struct assembly {
  specfem::compute::mesh mesh;
  specfem::compute::partial_derivatives partial_derivatives;
  specfem::compute::properties properties;
  specfem::compute::sources sources;
  specfem::compute::receivers receivers;
  specfem::compute::boundaries boundaries;
  specfem::compute::coupled_interfaces coupled_interfaces;
  specfem::compute::fields fields;

  assembly(
      const specfem::mesh::mesh &mesh,
      const specfem::quadrature::quadratures &quadratures,
      const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
      const std::vector<std::shared_ptr<specfem::receivers::receiver> >
          &receivers,
      const std::vector<specfem::enums::seismogram::type> &stypes,
      const type_real t0, const type_real dt, const int max_timesteps,
      const int max_sig_step, const specfem::simulation::type simulation);
};

} // namespace compute
} // namespace specfem

#endif
