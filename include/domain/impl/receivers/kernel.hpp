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

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
class receiver_kernel {

public:
  receiver_kernel() = default;

  receiver_kernel(const specfem::compute::assembly &assembly);

  void compute_seismograms(const int &isig_step) const;

private:
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  IndexViewType elements;
  specfem::compute::assembly assembly;
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
      receiver_kernel_index_mapping;
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem

#endif // _DOMAIN_IMPL_RECEIVERS_KERNEL_HPP
