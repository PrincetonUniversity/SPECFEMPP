#ifndef _SPECFEM_COMPUTE_KERNELS_HPP_
#define _SPECFEM_COMPUTE_KERNELS_HPP_

#include "enumerations/medium.hpp"
#include "impl/material_kernels.hpp"
#include "mesh/materials/interface.hpp"
#include "point/kernels.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
struct kernels {
  int nspec;
  int ngllz;
  int ngllx;
  specfem::kokkos::DeviceView1d<specfem::element::medium_tag> element_types;
  specfem::kokkos::DeviceView1d<specfem::element::property_tag>
      element_property; ///< Element properties
  specfem::kokkos::HostMirror1d<specfem::element::medium_tag> h_element_types;
  specfem::kokkos::HostMirror1d<specfem::element::property_tag>
      h_element_property; ///< Element properties
  specfem::kokkos::DeviceView1d<int> property_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_property_index_mapping;

  specfem::compute::impl::kernels::material_kernels<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>
      elastic_isotropic;

  specfem::compute::impl::kernels::material_kernels<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>
      acoustic_isotropic;

  kernels() = default;

  kernels(const int nspec, const int ngllz, const int ngllx,
          const specfem::mesh::materials &materials);

  template <specfem::element::medium_tag type,
            specfem::element::property_tag property>
  KOKKOS_FUNCTION void
  update_kernels(const int ispec, const int iz, const int ix,
                 const specfem::point::kernels<type, property> &kernels) const {
    const int index = property_index_mapping(ispec);

    if constexpr ((type == specfem::element::medium_tag::elastic) &&
                  (property == specfem::element::property_tag::isotropic)) {
      return elastic_isotropic.update_kernels(index, iz, ix, kernels);
    } else if constexpr ((type == specfem::element::medium_tag::acoustic) &&
                         (property ==
                          specfem::element::property_tag::isotropic)) {
      return acoustic_isotropic.update_kernels(index, iz, ix, kernels);
    } else {
      static_assert("Material type not implemented");
    }
  }

  template <specfem::element::medium_tag type,
            specfem::element::property_tag property>
  KOKKOS_FUNCTION auto get_kernels() const {
    if constexpr ((type == specfem::element::medium_tag::elastic) &&
                  (property == specfem::element::property_tag::isotropic)) {
      return elastic_isotropic;
    } else if constexpr ((type == specfem::element::medium_tag::acoustic) &&
                         (property ==
                          specfem::element::property_tag::isotropic)) {
      return acoustic_isotropic;
    } else {
      static_assert("Material type not implemented");
    }
  }

  void sync_views() {
    Kokkos::deep_copy(h_element_types, element_types);
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
    elastic_isotropic.sync_views();
    acoustic_isotropic.sync_views();
  }
};
} // namespace compute
} // namespace specfem

#endif /* _SPECFEM_COMPUTE_KERNELS_HPP_ */
