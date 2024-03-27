#ifndef _COMPUTE_SOURCES_IMPL_SOURCE_MEDIUM_HPP
#define _COMPUTE_SOURCES_IMPL_SOURCE_MEDIUM_HPP

#include "compute/compute_mesh.hpp"
#include "source/source.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
namespace impl {
namespace sources {
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag Medium>
struct source_medium {
  using medium_type = specfem::medium::medium<Dimension, Medium>;

  source_medium() = default;

  source_medium(
      const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties, const type_real t0,
      const type_real dt, const int nsteps);

  specfem::kokkos::DeviceView1d<int> source_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_source_index_mapping;
  specfem::kokkos::DeviceView3d<type_real> source_time_function;
  specfem::kokkos::HostMirror3d<type_real> h_source_time_function;
  specfem::kokkos::DeviceView4d<type_real> source_array;
  specfem::kokkos::HostMirror4d<type_real> h_source_array;
};
} // namespace sources
} // namespace impl
} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_SOURCES_IMPL_SOURCE_MEDIUM_HPP */
