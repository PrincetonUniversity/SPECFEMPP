#pragma once

#include "compute/element_types/element_types.hpp"
#include "compute/impl/value_containers.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "medium/material.hpp"
#include "medium/material_properties.hpp"
#include "medium/properties_container.hpp"
#include "point/coordinates.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

namespace specfem {
namespace compute {

/**
 * @brief Material properties at every quadrature point in the finite element
 * mesh
 *
 */
struct properties
    : public impl::value_containers<specfem::medium::material_properties> {
  /**
   * @name Constructors
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  properties() = default;

  /**
   * @brief Construct a new properties object from mesh information
   *
   * @param nspec Number of spectral elements
   * @param ngllz Number of quadrature points in z direction
   * @param ngllx Number of quadrature points in x direction
   * @param mapping Mapping of spectral element index from mesh to assembly
   * @param tags Element Tags for every spectral element
   * @param materials Material properties for every spectral element
   * @param has_gll_model Whether a GLL model is present (skip material property
   * assignment if true)
   */
  properties(
      const int nspec, const int ngllz, const int ngllx,
      const specfem::compute::element_types &element_types,
      const specfem::mesh::materials<specfem::dimension::type::dim2> &materials,
      bool has_gll_model);

  ///@}

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    impl::value_containers<
        specfem::medium::material_properties>::copy_to_host();
  }

  void copy_to_device() {
    impl::value_containers<
        specfem::medium::material_properties>::copy_to_device();
  }
};

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
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const IndexType &lcoord,
               const specfem::compute::properties &properties,
               PointPropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  IndexType l_index = lcoord;

  const int index = properties.property_index_mapping(ispec);

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionType = PointPropertiesType::dimension;

  static_assert(DimensionType == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  properties.get_container<MediumTag, PropertyTag>().load_device_properties(
      l_index, point_properties);
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
void load_on_host(const IndexType &lcoord,
                  const specfem::compute::properties &properties,
                  PointPropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  IndexType l_index = lcoord;

  const int index = properties.h_property_index_mapping(ispec);

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionType = PointPropertiesType::dimension;

  static_assert(DimensionType == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  properties.get_container<MediumTag, PropertyTag>().load_host_properties(
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
void store_on_host(const IndexType &lcoord,
                   const specfem::compute::properties &properties,
                   const PointPropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  const int index = properties.h_property_index_mapping(ispec);

  IndexType l_index = lcoord;

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionType = PointPropertiesType::dimension;

  static_assert(DimensionType == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  properties.get_container<MediumTag, PropertyTag>().assign(l_index,
                                                            point_properties);
}
} // namespace compute
} // namespace specfem
