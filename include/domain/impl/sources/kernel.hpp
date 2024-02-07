#ifndef _DOMAIN_IMPL_SOURCES_KERNEL_HPP
#define _DOMAIN_IMPL_SOURCES_KERNEL_HPP

#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/interface.hpp"
#include "domain/impl/sources/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

template <class medium, class qp_type, class property> class source_kernel {
public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium_type = medium;
  using property_type = property;
  using quadrature_point_type = qp_type;

  source_kernel() = default;
  source_kernel(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos::HostView1d<int> h_source_kernel_index_mapping,
      const specfem::kokkos::HostView1d<int> h_source_mapping,
      const quadrature_point_type quadrature_points);

  void compute_source_interaction(const int timestep) const;

private:
  int nsources;
  specfem::compute::points points;
  specfem::compute::quadrature quadrature;
  specfem::kokkos::DeviceView1d<int> source_kernel_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_source_kernel_index_mapping;
  specfem::kokkos::DeviceView1d<int> source_mapping;
  Kokkos::View<int * [specfem::enums::element::ntypes], Kokkos::LayoutLeft,
               specfem::kokkos::DevMemSpace>
      global_index_mapping;
  specfem::compute::properties properties;
  specfem::compute::impl::field_impl<medium_type> field;
  specfem::compute::sources sources;
  quadrature_point_type quadrature_points;
  specfem::domain::impl::sources::source<dimension, medium_type,
                                         quadrature_point_type, property_type>
      source;
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // _DOMAIN_IMPL_SOURCES_KERNEL_HPP
