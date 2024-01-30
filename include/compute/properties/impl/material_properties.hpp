#ifndef _COMPUTE_PROPERTIES_IMPL_MATERIAL_PROPERTIES_HPP_
#define _COMPUTE_PROPERTIES_IMPL_MATERIAL_PROPERTIES_HPP_

#include "mesh/materials/interface.hpp"
#include "properties_container.hpp"

namespace specfem {
namespace compute {
namespace impl {
namespace properties {
template <specfem::enums::element::type type,
          specfem::enums::element::property_tag property>
struct material_property
    : public specfem::compute::impl::properties::properties_container<
          type, property> {
  constexpr static auto value_type = type;
  constexpr static auto property_type = property;

  material_property() = default;

  material_property(
      const int nspec, const int n_element, const int ngllz, const int ngllx,
      const specfem::mesh::materials &materials,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : specfem::compute::impl::properties::properties_container<type,
                                                                 property>(
            n_element, ngllz, ngllx) {

    int count = 0;
    for (int ispec = 0; ispec < nspec; ++ispec) {
      const auto material_specification =
          materials.material_index_mapping(ispec);
      const int index = material_specification.index;

      if ((material_specification.type == type) &&
          (material_specification.property == property)) {
        property_index_mapping(ispec) = count;
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int ix = 0; ix < ngllx; ++ix) {
            // Get the material at index from mesh::materials
            auto material =
                std::get<specfem::material::material<type, property> >(
                    materials[ispec]);

            // Assign the material property to the property container
            auto point_property = material.get_properties();
            this->assign(count, iz, ix, point_property);
          }
        }
        count++;
      }
    }

    assert(count == n_element);

    this->copy_to_device();

    return;
  }
};
} // namespace properties
} // namespace impl
} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_PROPERTIES_IMPL_MATERIAL_PROPERTIES_HPP_ */
