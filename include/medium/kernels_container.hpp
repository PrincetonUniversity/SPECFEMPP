#pragma once

#include "impl/medium_data.hpp"

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int N>
struct impl_kernels_container
    : public impl::medium_data<MediumTag, PropertyTag, N> {
  using base_type = impl::medium_data<MediumTag, PropertyTag, N>;
  using base_type::base_type;

  impl_kernels_container(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int ngllz, const int ngllx,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : impl_kernels_container(elements.extent(0), ngllz, ngllx) {

    const int nelement = elements.extent(0);
    int count = 0;
    for (int i = 0; i < nelement; ++i) {
      const int ispec = elements(i);
      property_index_mapping(ispec) = count;
      count++;
    }
  }
};

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
class kernels_container;

} // namespace medium
} // namespace specfem

// Including the template specializations here so that kernels_container is
// an interface to the compute/kernels module
#include "dim2/acoustic/isotropic/kernels_container.hpp"
#include "dim2/elastic/anisotropic/kernels_container.hpp"
#include "dim2/elastic/isotropic/kernels_container.hpp"
