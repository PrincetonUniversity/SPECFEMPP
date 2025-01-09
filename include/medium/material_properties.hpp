#pragma once

#include "compute/compute_mesh.hpp"
#include "enumerations/interface.hpp"
#include "medium/properties_container.hpp"
#include "mesh/materials/materials.hpp"
#include "mesh/tags/tags.hpp"

namespace specfem {
namespace medium {

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
struct material_properties
    : public specfem::medium::properties_container<type, property> {
  constexpr static auto value_type = type;
  constexpr static auto property_type = property;
  constexpr static auto dimension = specfem::dimension::type::dim2;

  material_properties() = default;

  material_properties(
      const int nspec, const int n_element, const int ngllz, const int ngllx,
      const specfem::compute::mesh_to_compute_mapping &mapping,
      const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
      const specfem::mesh::materials &materials, const bool &has_gll_model,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : specfem::medium::properties_container<type, property>(n_element, ngllz,
                                                              ngllx) {

    int count = 0;
    for (int ispec = 0; ispec < nspec; ++ispec) {
      const int ispec_mesh = mapping.compute_to_mesh(ispec);
      const auto &tag = tags.tags_container(ispec_mesh);

      if ((tag.medium_tag == type) && (tag.property_tag == property)) {
        property_index_mapping(ispec) = count;
        if (!has_gll_model) {
          for (int iz = 0; iz < ngllz; ++iz) {
            for (int ix = 0; ix < ngllx; ++ix) {
              // Get the material at index from mesh::materials
              auto material =
                  std::get<specfem::medium::material<type, property> >(
                      materials[ispec_mesh]);

              // Assign the material property to the property container
              auto point_property = material.get_properties();
              this->assign(specfem::point::index<dimension>(count, iz, ix),
                           point_property);
            }
          }
        }
        count++;
      }
    }

    assert(count == n_element);

    if (!has_gll_model) {
      this->copy_to_device();
    }

    return;
  }
};

} // namespace medium
} // namespace specfem
