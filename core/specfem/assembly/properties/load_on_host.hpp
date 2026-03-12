#pragma once

#include "specfem/assembly/properties.hpp"

namespace specfem::assembly {

/**
 * @brief Load material properties at a quadrature point on host memory.
 *
 * This function retrieves material properties from the assembly properties
 * container and loads them into a point properties structure for use in
 * host-side computations.
 *
 * The function is optimized for host execution and supports both regular and
 * SIMD operations depending on the template parameters. SIMD compatibility is
 * enforced through SFINAE.
 *
 * @ingroup PropertiesDataAccess
 *
 * @tparam PointPropertiesType Material properties type (e.g.,
 * specfem::point::properties) Must specify medium_tag, property_tag, and
 * dimension_tag
 * @tparam IndexType Index type for quadrature point location (e.g.,
 * specfem::point::index) Must have compatible SIMD settings with
 * PointPropertiesType
 *
 * @param lcoord Local coordinate index specifying element and quadrature point
 * location
 * @param properties Assembly properties container holding all material data
 * @param point_properties Output structure to receive the loaded material
 * properties
 *
 * @code
 * // Load elastic isotropic properties at a quadrature point
 * specfem::point::index<...> coord(ispec, iz, ix);
 * specfem::point::properties<...> props;
 *
 * load_on_host(coord, assembly.properties, props);
 * @endcode
 *
 */
template <typename PointPropertiesType, typename IndexType,
          typename std::enable_if_t<IndexType::using_simd ==
                                        PointPropertiesType::simd::using_simd,
                                    int> = 0>
void load_on_host(
    const IndexType &lcoord,
    const specfem::assembly::properties<PointPropertiesType::dimension_tag>
        &properties,
    PointPropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  IndexType l_index = lcoord;

  const int index = properties.h_property_index_mapping(ispec);

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;

  properties.template get_container<MediumTag, PropertyTag>().load_host_values(
      l_index, point_properties);
}

} // namespace specfem::assembly
