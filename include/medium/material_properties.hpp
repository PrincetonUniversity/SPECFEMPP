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
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int ngllz, const int ngllx,
      const specfem::mesh::materials &materials, const bool has_gll_model,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : specfem::medium::properties_container<type, property>(
            elements.extent(0), ngllz, ngllx) {

    const int nelement = elements.extent(0);
    int count = 0;
    for (int i = 0; i < nelement; ++i) {
      const int ispec = elements(i);
      property_index_mapping(ispec) = count;
      if (!has_gll_model) {
        // Get the material at index from mesh::materials
        const auto material =
            std::get<specfem::medium::material<type, property> >(
                materials[ispec]);

        // Assign the material property to the property container
        const auto point_property = material.get_properties();
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int ix = 0; ix < ngllx; ++ix) {
            this->assign(specfem::point::index<dimension>(count, iz, ix),
                         point_property);
          }
        }
      }
      count++;
    }

    if (!has_gll_model) {
      this->copy_to_device();
    }

    return;
  }
};

} // namespace medium
} // namespace specfem
