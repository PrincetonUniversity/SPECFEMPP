#pragma once

#include "specfem/assembly/properties.hpp"

namespace specfem::assembly {

/**
 * @brief Store material properties at a quadrature point in host memory.
 *
 * This function writes material properties from a point properties structure
 * into the assembly properties container's host memory storage.
 *
 * The function is designed for host-side operations such as initialization,
 * post-processing, debugging, or I/O operations. It supports both regular
 * and SIMD operations with SIMD compatibility enforced through SFINAE.
 * This is the counterpart to load_on_host for bidirectional data access.
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
 * @param point_properties Input structure containing material properties to
 * store
 * @param properties Assembly properties container to receive the material data
 *
 * @code
 * // Store modified elastic properties back to host memory
 * specfem::point::index<...> coord(ispec, iz, ix);
 * specfem::point::properties<...> props;
 *
 * // Modify properties (e.g., after inversion or processing)
 * props.rho = updated_density;
 * props.lambda = updated_lambda;
 * props.mu = updated_mu;
 *
 * store_on_host(coord, props, assembly.properties);
 * @endcode
 */
template <typename PointPropertiesType, typename IndexType,
          typename std::enable_if_t<IndexType::using_simd ==
                                        PointPropertiesType::simd::using_simd,
                                    int> = 0>
void store_on_host(
    const IndexType &lcoord, const PointPropertiesType &point_properties,
    const specfem::assembly::properties<PointPropertiesType::dimension_tag>
        &properties) {
  const int ispec = lcoord.ispec;

  const int index = properties.h_property_index_mapping(ispec);

  IndexType l_index = lcoord;

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;

  properties.template get_container<MediumTag, PropertyTag>().store_host_values(
      l_index, point_properties);
}

} // namespace specfem::assembly
