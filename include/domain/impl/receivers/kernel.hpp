#ifndef _DOMAIN_IMPL_RECEIVERS_KERNEL_HPP
#define _DOMAIN_IMPL_RECEIVERS_KERNEL_HPP

#include "domain/impl/receivers/acoustic/interface.hpp"
#include "domain/impl/receivers/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename qp_type>
class receiver_kernel {
public:
  using dimension = specfem::dimension::dimension<DimensionType>;
  using medium_type =
      specfem::medium::medium<DimensionType, MediumTag, PropertyTag>;
  using quadrature_points_type = qp_type;

  receiver_kernel() = default;

  receiver_kernel(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos::HostView1d<int> h_receiver_kernel_index_mapping,
      const specfem::kokkos::HostView1d<int> h_receiver_mapping,
      const quadrature_points_type &quadrature_points);

  void compute_seismograms(const int &isig_step) const;

private:
  int nreceivers;
  int nseismograms;
  specfem::compute::points points;
  specfem::compute::quadrature quadrature;
  specfem::compute::partial_derivatives partial_derivatives;
  specfem::compute::properties properties;
  specfem::kokkos::DeviceView1d<int> receiver_kernel_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_receiver_kernel_index_mapping;
  specfem::kokkos::DeviceView1d<int> receiver_mapping;
  specfem::compute::impl::field_impl<DimensionType, MediumTag> field;
  Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
               specfem::kokkos::DevMemSpace>
      global_index_mapping;
  specfem::compute::receivers receivers;
  quadrature_points_type quadrature_points;

  specfem::domain::impl::receivers::receiver<
      DimensionType, MediumTag, PropertyTag, quadrature_points_type>
      receiver;
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // _DOMAIN_IMPL_RECEIVERS_KERNEL_HPP
