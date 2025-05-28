#pragma once

#include "dim2/acoustic/isotropic/kernels_container.hpp"
#include "dim2/elastic/anisotropic/kernels_container.hpp"
#include "dim2/elastic/isotropic/kernels_container.hpp"
#include "dim2/poroelastic/isotropic/kernels_container.hpp"
#include "impl/accessor.hpp"
#include "impl/data_container.hpp"

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct kernels_container
    : public kernels::data_container<MediumTag, PropertyTag>,
      public impl::Accessor<kernels_container<MediumTag, PropertyTag> > {

  using base_type = kernels::data_container<MediumTag, PropertyTag>;
  using base_type::base_type;
  kernels_container() = default;

  kernels_container(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int ngllz, const int ngllx,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : base_type(elements.extent(0), ngllz, ngllx) {
    const int nelement = elements.extent(0);
    int count = 0;
    for (int i = 0; i < nelement; ++i) {
      const int ispec = elements(i);
      property_index_mapping(ispec) = count;
      count++;
    }
  }
};
} // namespace medium
} // namespace specfem
