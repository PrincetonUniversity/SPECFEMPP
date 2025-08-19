#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

/**
 * @brief Material properties at every quadrature point in the finite element
 * mesh
 *
 */
template <specfem::dimension::type DimensionTag> struct properties;

/**
 * @defgroup ComputePropertiesDataAccess
 */

/**
 * @brief Load the material properties at a given quadrature point on the device
 *
 * @ingroup ComputePropertiesDataAccess
 *
 * @tparam PointPropertiesType Point properties type. Needs to be of @ref
 * specfem::point::properties
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param lcoord Index of the quadrature point
 * @param properties Material properties container
 * @param point_properties Material properties at the given quadrature point
 * (output)
 */
template <typename PointPropertiesType, typename IndexType,
          typename std::enable_if_t<IndexType::using_simd ==
                                        PointPropertiesType::simd::using_simd,
                                    int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
    const IndexType &lcoord,
    const specfem::assembly::properties<PointPropertiesType::dimension_tag>
        &properties,
    PointPropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  IndexType l_index = lcoord;

  const int index = properties.property_index_mapping(ispec);

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;

  properties.template get_container<MediumTag, PropertyTag>()
      .load_device_values(l_index, point_properties);
}

/**
 * @brief Store the material properties at a given quadrature point on the
 * device
 *
 * @ingroup ComputePropertiesDataAccess
 *
 * @tparam PointPropertiesType Point properties type. Needs to be of @ref
 * specfem::point::properties
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param lcoord Index of the quadrature point
 * @param properties Material properties container
 * @param point_properties Material properties at the given quadrature point
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

/**
 * @brief Store the material properties at a given quadrature point on the host
 *
 * @ingroup ComputePropertiesDataAccess
 *
 * @tparam PointPropertiesType Point properties type. Needs to be of @ref
 * specfem::point::properties
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param lcoord Index of the quadrature point
 * @param properties Material properties container
 * @param point_properties Material properties at the given quadrature point
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

#include "properties/dim2/properties.hpp"
#include "properties/dim3/properties.hpp"
