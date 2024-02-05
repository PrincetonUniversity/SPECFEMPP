#ifndef _COMPUTE_PROPERTIES_HPP
#define _COMPUTE_PROPERTIES_HPP

#include "enumerations/specfem_enums.hpp"
#include "impl/material_properties.hpp"
#include "impl/properties_container.hpp"
#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

namespace specfem {
namespace compute {

/**
 * @brief Material properties stored at every quadrature point
 *
 */
struct properties {

  int nspec; ///< total number of spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension
  specfem::kokkos::DeviceView1d<int> property_index_mapping;   ///< Mapping of
                                                               ///< spectral
                                                               ///< element to
                                                               ///< material
                                                               ///< properties
  specfem::kokkos::HostMirror1d<int> h_property_index_mapping; ///< Mapping of
                                                               ///< spectral
                                                               ///< element to
                                                               ///< material
                                                               ///< properties
  specfem::kokkos::DeviceView1d<specfem::enums::element::type>
      element_types; ///< Element types
  specfem::kokkos::DeviceView1d<specfem::enums::element::property_tag>
      element_property; ///< Element properties
  specfem::kokkos::HostMirror1d<specfem::enums::element::type>
      h_element_types; ///< Element types
  specfem::kokkos::HostMirror1d<specfem::enums::element::property_tag>
      h_element_property; ///< Element properties
  specfem::compute::impl::properties::material_property<
      specfem::enums::element::type::elastic,
      specfem::enums::element::property_tag::isotropic>
      elastic_isotropic;
  specfem::compute::impl::properties::material_property<
      specfem::enums::element::type::acoustic,
      specfem::enums::element::property_tag::isotropic>
      acoustic_isotropic;

  template <specfem::enums::element::type type,
            specfem::enums::element::property_tag property, typename ExecSpace>
  KOKKOS_FUNCTION specfem::point::properties<type, property>
  load_properties(const int ispec, const int iz, const int ix) const {
    const specfem::compute::impl::properties::properties_container<type,
                                                                   property>
        container = [&]() {
          if constexpr ((type == specfem::enums::element::type::elastic) &&
                        (property ==
                         specfem::enums::element::property_tag::isotropic)) {
            return elastic_isotropic;
          } else if constexpr ((type ==
                                specfem::enums::element::type::acoustic) &&
                               (property == specfem::enums::element::
                                                property_tag::isotropic)) {
            return acoustic_isotropic;
          } else {
            static_assert("Material type not implemented");
          }
        }();

    const int index = [&]() {
      if constexpr (std::is_same_v<ExecSpace, specfem::kokkos::DevExecSpace>) {
        return property_index_mapping(ispec);
      } else {
        return h_property_index_mapping(ispec);
      }
    }();

    return container.template load_properties<ExecSpace>(index, iz, ix);
  }

  properties() = default;

  properties(const int nspec, const int ngllz, const int ngllx,
             const specfem::mesh::materials &materials);
};

} // namespace compute
} // namespace specfem

#endif
