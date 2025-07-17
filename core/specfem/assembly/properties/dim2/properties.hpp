#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "medium/material.hpp"
#include "medium/properties_container.hpp"
#include "specfem/assembly/assembly/value_containers.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

namespace specfem::assembly {

template <>
struct properties<specfem::dimension::type::dim2>
    : public impl::value_containers<specfem::dimension::type::dim2,
                                    specfem::medium::properties_container> {
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
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::mesh::materials<dimension_tag> &materials,
      bool has_gll_model);

  ///@}

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    impl::value_containers<
        dimension_tag, specfem::medium::properties_container>::copy_to_host();
  }

  void copy_to_device() {
    impl::value_containers<
        dimension_tag, specfem::medium::properties_container>::copy_to_device();
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
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
    const IndexType &lcoord,
    const specfem::assembly::properties<specfem::dimension::type::dim2>
        &properties,
    PointPropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  IndexType l_index = lcoord;

  const int index = properties.property_index_mapping(ispec);

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;

  static_assert(DimensionTag == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  properties.get_container<MediumTag, PropertyTag>().load_device_values(
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
void load_on_host(
    const IndexType &lcoord,
    const specfem::assembly::properties<specfem::dimension::type::dim2>
        &properties,
    PointPropertiesType &point_properties) {
  const int ispec = lcoord.ispec;

  IndexType l_index = lcoord;

  const int index = properties.h_property_index_mapping(ispec);

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;

  static_assert(DimensionTag == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  properties.get_container<MediumTag, PropertyTag>().load_host_values(
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
    const specfem::assembly::properties<specfem::dimension::type::dim2>
        &properties) {
  const int ispec = lcoord.ispec;

  const int index = properties.h_property_index_mapping(ispec);

  IndexType l_index = lcoord;

  l_index.ispec = index;

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;

  static_assert(DimensionTag == specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  properties.get_container<MediumTag, PropertyTag>().store_host_values(
      l_index, point_properties);
}

template <typename IndexViewType, typename PointPropertiesType>
void max(const IndexViewType &ispecs,
         const specfem::assembly::properties<specfem::dimension::type::dim2>
             &properties,
         PointPropertiesType &point_properties) {

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;

  constexpr bool on_device =
      std::is_same<typename IndexViewType::execution_space,
                   Kokkos::DefaultExecutionSpace>::value;

  static_assert(PointPropertiesType::dimension_tag ==
                    specfem::dimension::type::dim2,
                "Only 2D properties are supported");

  IndexViewType local_ispecs("local_ispecs", ispecs.extent(0));

  const auto index_mapping = properties.get_property_index_mapping<on_device>();

  Kokkos::parallel_for(
      "local_work_items",
      Kokkos::RangePolicy<typename IndexViewType::execution_space>(
          0, ispecs.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        local_ispecs(i) =
            index_mapping(ispecs(i)); // Map the ispec to the property index
      });

  Kokkos::fence(); // Ensure the above parallel for is complete before
                   // proceeding

  properties.get_container<MediumTag, PropertyTag>().max(local_ispecs,
                                                         point_properties);

  // Note: The above call to max will perform the reduction on the device

  return;
}
} // namespace specfem::assembly
