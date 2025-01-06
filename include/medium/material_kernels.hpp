#pragma once

#include "compute/properties/properties.hpp"
#include "enumerations/interface.hpp"
#include "kernels_container.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
class material_kernels : public kernels_container<type, property> {
public:
  constexpr static auto value_type = type;
  constexpr static auto property_type = property;

  material_kernels() = default;

  material_kernels(
      const int nspec, const int n_element, const int ngllz, const int ngllx,
      const specfem::compute::mesh_to_compute_mapping &mapping,
      const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : specfem::compute::impl::kernels::kernels_container<value_type,
                                                           property_type>(
            n_element, ngllz, ngllx) {
    int count = 0;
    for (int ispec = 0; ispec < nspec; ++ispec) {
      const int ispec_mesh = mapping.compute_to_mesh(ispec);
      const auto &tag = tags.tags_container(ispec_mesh);
      if ((tag.medium_tag == type) && (tag.property_tag == property)) {
        property_index_mapping(ispec) = count;
        count++;
      }
    }

    assert(count == n_element);
  }
};
} // namespace medium
} // namespace specfem
