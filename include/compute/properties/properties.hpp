#ifndef _COMPUTE_PROPERTIES_HPP
#define _COMPUTE_PROPERTIES_HPP

#include "enumerations/specfem_enums.hpp"
#include "impl/material_properties.hpp"
#include "impl/properties_container.hpp"
#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "point/coordinates.hpp"
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
  specfem::kokkos::DeviceView1d<specfem::element::medium_tag>
      element_types; ///< Element types
  specfem::kokkos::DeviceView1d<specfem::element::property_tag>
      element_property; ///< Element properties
  specfem::kokkos::HostMirror1d<specfem::element::medium_tag>
      h_element_types; ///< Element types
  specfem::kokkos::HostMirror1d<specfem::element::property_tag>
      h_element_property; ///< Element properties

  specfem::compute::impl::properties::material_property<
      specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic>
      elastic_isotropic;
  specfem::compute::impl::properties::material_property<
      specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic>
      acoustic_isotropic;

  // template <specfem::element::medium_tag type,
  //           specfem::element::property_tag property>
  // KOKKOS_FUNCTION specfem::point::properties<type, property>
  // load_device_properties(const int ispec, const int iz, const int ix) const {
  //   const int index = property_index_mapping(ispec);

  //   if constexpr ((type == specfem::element::medium_tag::elastic) &&
  //                 (property == specfem::element::property_tag::isotropic)) {
  //     return elastic_isotropic.load_device_properties(index, iz, ix);
  //   } else if constexpr ((type == specfem::element::medium_tag::acoustic) &&
  //                        (property ==
  //                         specfem::element::property_tag::isotropic)) {
  //     return acoustic_isotropic.load_device_properties(index, iz, ix);
  //   } else {
  //     static_assert("Material type not implemented");
  //   }
  // }

  // template <specfem::element::medium_tag type,
  //           specfem::element::property_tag property>
  // specfem::point::properties<type, property>
  // load_host_properties(const int ispec, const int iz, const int ix) const {
  //   const int index = h_property_index_mapping(ispec);

  //   if constexpr ((type == specfem::element::medium_tag::elastic) &&
  //                 (property == specfem::element::property_tag::isotropic)) {
  //     return elastic_isotropic.load_host_properties(index, iz, ix);
  //   } else if constexpr ((type == specfem::element::medium_tag::acoustic) &&
  //                        (property ==
  //                         specfem::element::property_tag::isotropic)) {
  //     return acoustic_isotropic.load_host_properties(index, iz, ix);
  //   } else {
  //     static_assert("Material type not implemented");
  //   }
  // }

  properties() = default;

  properties(const int nspec, const int ngllz, const int ngllx,
             const specfem::mesh::materials &materials);
};

template <typename PropertiesType, typename IndexType>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const IndexType &lcoord,
               const specfem::compute::properties &properties,
               PropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  IndexType l_index = lcoord;

  const int index = properties.property_index_mapping(ispec);

  l_index.ispec = index;

  constexpr auto MediumTag = PropertiesType::medium_tag;
  constexpr auto PropertyTag = PropertiesType::property_tag;
  constexpr auto DimensionType = PropertiesType::dimension;

  static_assert(DimensionType == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    properties.elastic_isotropic.load_device_properties(l_index,
                                                        point_properties);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    properties.acoustic_isotropic.load_device_properties(l_index,
                                                         point_properties);
  } else {
    static_assert("Material type not implemented");
  }
}

template <typename PropertiesType, typename IndexType>
void load_on_host(const IndexType &lcoord,
                  const specfem::compute::properties &properties,
                  PropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  IndexType l_index = lcoord;

  const int index = properties.h_property_index_mapping(ispec);

  l_index.ispec = index;

  constexpr auto MediumTag = PropertiesType::medium_tag;
  constexpr auto PropertyTag = PropertiesType::property_tag;
  constexpr auto DimensionType = PropertiesType::dimension;

  static_assert(DimensionType == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    properties.elastic_isotropic.load_host_properties(l_index,
                                                      point_properties);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    properties.acoustic_isotropic.load_host_properties(l_index,
                                                       point_properties);
  } else {
    static_assert("Material type not implemented");
  }
}

template <typename PropertiesType, typename IndexType>
void store_on_host(const IndexType &lcoord,
                   const specfem::compute::properties &properties,
                   const PropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  const int index = properties.h_property_index_mapping(ispec);

  IndexType l_index = lcoord;

  l_index.ispec = index;

  constexpr auto MediumTag = PropertiesType::medium_tag;
  constexpr auto PropertyTag = PropertiesType::property_tag;
  constexpr auto DimensionType = PropertiesType::dimension;

  static_assert(DimensionType == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
                (PropertyTag == specfem::element::property_tag::isotropic)) {
    properties.elastic_isotropic.assign(l_index, point_properties);
  } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic) &&
                       (PropertyTag ==
                        specfem::element::property_tag::isotropic)) {
    properties.acoustic_isotropic.assign(l_index, point_properties);
  } else {
    static_assert("Material type not implemented");
  }
}

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag>
// KOKKOS_FUNCTION void
// load_on_device(const specfem::point::index &lcoord,
//                const specfem::compute::properties &properties,
//                specfem::point::properties<DimensionType, MediumTag,
//                PropertyTag>
//                    &point_properties) {
//   const int ispec = lcoord.ispec;
//   const int iz = lcoord.iz;
//   const int ix = lcoord.ix;
//   const int index = properties.property_index_mapping(ispec);

//   static_assert(DimensionType == specfem::dimension::type::dim2,
//                 "Only 2D properties are supported");

//   if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
//                 (PropertyTag == specfem::element::property_tag::isotropic)) {
//     properties.elastic_isotropic.load_device_properties(index, iz, ix,
//                                                         point_properties);
//   } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic)
//   &&
//                        (PropertyTag ==
//                         specfem::element::property_tag::isotropic)) {
//     properties.acoustic_isotropic.load_device_properties(index, iz, ix,
//                                                          point_properties);
//   } else {
//     static_assert("Material type not implemented");
//   }

//   return;
// }

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag>
// void load_on_host(const specfem::point::index &lcoord,
//                   const specfem::compute::properties &properties,
//                   specfem::point::properties<DimensionType, MediumTag,
//                                              PropertyTag> &point_properties)
//                                              {

//   const int ispec = lcoord.ispec;
//   const int iz = lcoord.iz;
//   const int ix = lcoord.ix;
//   const int index = properties.h_property_index_mapping(ispec);

//   static_assert(DimensionType == specfem::dimension::type::dim2,
//                 "Only 2D properties are supported");

//   if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
//                 (PropertyTag == specfem::element::property_tag::isotropic)) {
//     properties.elastic_isotropic.load_host_properties(index, iz, ix,
//                                                       point_properties);
//   } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic)
//   &&
//                        (PropertyTag ==
//                         specfem::element::property_tag::isotropic)) {
//     properties.acoustic_isotropic.load_host_properties(index, iz, ix,
//                                                        point_properties);
//   } else {
//     static_assert("Material type not implemented");
//   }

//   return;
// }

// template <specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag>
// void store_on_host(
//     const specfem::point::index &lcoord,
//     const specfem::compute::properties &properties,
//     const specfem::point::properties<DimensionType, MediumTag, PropertyTag>
//         &point_properties) {
//   const int ispec = lcoord.ispec;
//   const int iz = lcoord.iz;
//   const int ix = lcoord.ix;
//   const int index = properties.h_property_index_mapping(ispec);

//   static_assert(DimensionType == specfem::dimension::type::dim2,
//                 "Only 2D properties are supported");

//   if constexpr ((MediumTag == specfem::element::medium_tag::elastic) &&
//                 (PropertyTag == specfem::element::property_tag::isotropic)) {
//     properties.elastic_isotropic.assign(index, iz, ix, point_properties);
//   } else if constexpr ((MediumTag == specfem::element::medium_tag::acoustic)
//   &&
//                        (PropertyTag ==
//                         specfem::element::property_tag::isotropic)) {
//     properties.acoustic_isotropic.assign(index, iz, ix, point_properties);
//   } else {
//     static_assert("Material type not implemented");
//   }

//   return;
// }

} // namespace compute
} // namespace specfem

#endif
