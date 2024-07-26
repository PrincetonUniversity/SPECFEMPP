#ifndef _SPECFEM_COMPUTE_KERNELS_IMPL_MATERIAL_KERNELS_HPP_
#define _SPECFEM_COMPUTE_KERNELS_IMPL_MATERIAL_KERNELS_HPP_

#include "compute/properties/properties.hpp"
#include "enumerations/medium.hpp"
#include "kernels_container.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
namespace impl {
namespace kernels {
template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
class material_kernels : public kernels_container<type, property> {
public:
  constexpr static auto value_type = type;
  constexpr static auto property_type = property;

  material_kernels() = default;

  material_kernels(const int nspec, const int n_element, const int ngllz,
                   const int ngllx,
                   const specfem::compute::properties &properties,
                   specfem::kokkos::HostView1d<int> property_index_mapping)
      : specfem::compute::impl::kernels::kernels_container<value_type,
                                                           property_type>(
            n_element, ngllz, ngllx) {
    int count = 0;
    for (int ispec = 0; ispec < nspec; ++ispec) {
      const auto medium_tag = properties.h_element_types(ispec);
      const auto property_tag = properties.h_element_property(ispec);
      if ((medium_tag == type) && (property_tag == property)) {
        property_index_mapping(ispec) = count;
        count++;
      }
    }

    assert(count == n_element);
  }
};
} // namespace kernels
} // namespace impl
} // namespace compute
} // namespace specfem

#endif /* _SPECFEM_COMPUTE_KERNELS_IMPL_MATERIAL_KERNELS_HPP_ */
