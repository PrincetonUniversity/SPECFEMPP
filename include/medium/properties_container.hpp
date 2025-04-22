#pragma once

#include "dim2/acoustic/isotropic/properties_container.hpp"
#include "dim2/elastic/anisotropic/properties_container.hpp"
#include "dim2/elastic/isotropic/properties_container.hpp"
#include "dim2/poroelastic/isotropic/properties_container.hpp"
#include "enumerations/medium.hpp"
#include "impl/accessor.hpp"
#include "impl/data_container.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct properties_container
    : public properties::data_container<MediumTag, PropertyTag>,
      public impl::Accessor<properties_container<MediumTag, PropertyTag> > {

  using base_type = properties::data_container<MediumTag, PropertyTag>;
  using base_type::base_type;

  properties_container() = default;

  properties_container(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const specfem::compute::mesh_to_compute_mapping &mapping, const int ngllz,
      const int ngllx,
      const specfem::mesh::materials<specfem::dimension::type::dim2> &materials,
      const bool has_gll_model,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : base_type(elements.extent(0), ngllz, ngllx) {

    const int nelement = elements.extent(0);
    int count = 0;
    for (int i = 0; i < nelement; ++i) {
      const int ispec = elements(i);
      const int mesh_ispec = mapping.compute_to_mesh(ispec);
      property_index_mapping(ispec) = count;
      if (!has_gll_model) {
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int ix = 0; ix < ngllx; ++ix) {
            // Get the material at index from mesh::materials
            auto material =
                std::get<specfem::medium::material<base_type::medium_tag,
                                                   base_type::property_tag> >(
                    materials[mesh_ispec]);

            // Assign the material property to the property container
            auto point_property = material.get_properties();
            this->store_host_values(
                specfem::point::index<base_type::dimension>(count, iz, ix),
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

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_device_values(const IndexType &index, PointValues &values) const = delete;

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_host_values(const IndexType &index, PointValues &values) const = delete;
};

} // namespace medium
} // namespace specfem
