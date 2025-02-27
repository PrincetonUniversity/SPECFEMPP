#pragma once

#include "impl/containers.hpp"

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int N>
struct impl_properties_container
    : public impl::medium_container<MediumTag, PropertyTag, N> {
  using base_type = impl::medium_container<MediumTag, PropertyTag, N>;
  using base_type::base_type;

  impl_kernels_container(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int ngllz, const int ngllx,
      const specfem::mesh::materials &materials, const bool has_gll_model,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : impl_kernels_container(elements.extent(0), ngllz, ngllx) {

    const int nelement = elements.extent(0);
    int count = 0;
    for (int i = 0; i < nelement; ++i) {
      const int ispec = elements(i);
      property_index_mapping(ispec) = count;
      if (!has_gll_model) {
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int ix = 0; ix < ngllx; ++ix) {
            // Get the material at index from mesh::materials
            auto material =
                std::get<specfem::medium::material<base_type::medium_tag,
                                                   base_type::property_tag> >(
                    materials[ispec]);

            // Assign the material property to the property container
            auto point_property = material.get_properties();
            this->assign(
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
