#pragma once

#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @defgroup SourceDataAccess Source Data Access Functions
 */

/**
 * @brief Load source data for device-based computations
 * @ingroup SourceDataAccess
 *
 * @tparam IndexType Point index type (non-SIMD)
 * @tparam PointSourceType Point source type matching medium and wavefield tags
 * @tparam DimensionTag Spatial dimension (deduced from sources parameter)
 * @param index Spectral element index containing source location information
 * @param sources Source assembly with current timestep configuration
 * @param point_source [out] Output structure populated with source data
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 */
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag DimensionTag>
KOKKOS_INLINE_FUNCTION void load_on_device(
    const IndexType index,
    const specfem::assembly::sources<DimensionTag> &sources,
    PointSourceType &point_source) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when loading sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag == DimensionTag,
                "PointSourceType dimension_tag must match sources dimension_tag");

  static_assert(IndexType::dimension_tag == DimensionTag,
                "IndexType dimension_tag must match sources dimension_tag");

#ifndef NDEBUG
  const int isource = index.imap;

  if (index.ispec >= sources.nspec) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }
  if (sources.medium_types(isource) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }
  if (sources.wavefield_types(isource) != PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  using MediumTags = specfem::tags::Tags<PointSourceType::dimension_tag,
                                         PointSourceType::medium_tag>;

  sources.source_by_medium.template get<MediumTags>().load_on_device(
      sources.timestep, index, point_source);
}

/**
 * @brief Load source data for host-based computations
 * @ingroup SourceDataAccess
 *
 * @tparam IndexType Point index type (non-SIMD)
 * @tparam PointSourceType Point source type matching medium and wavefield tags
 * @tparam DimensionTag Spatial dimension (deduced from sources parameter)
 * @param index Spectral element index containing source location information
 * @param sources Source assembly with current timestep configuration
 * @param point_source [out] Output structure populated with source data
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 */
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag DimensionTag>
void load_on_host(
    const IndexType index,
    const specfem::assembly::sources<DimensionTag> &sources,
    PointSourceType &point_source) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when loading sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag == DimensionTag,
                "PointSourceType dimension_tag must match sources dimension_tag");

  static_assert(IndexType::dimension_tag == DimensionTag,
                "IndexType dimension_tag must match sources dimension_tag");

#ifndef NDEBUG
  const int isource = index.imap;

  if ((index.ispec < 0) || (sources.nspec <= index.ispec)) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }
  if (sources.h_medium_types(isource) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }
  if (sources.h_wavefield_types(isource) != PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  using MediumTags = specfem::tags::Tags<PointSourceType::dimension_tag,
                                         PointSourceType::medium_tag>;

  sources.source_by_medium.template get<MediumTags>().load_on_host(
      sources.timestep, index, point_source);
}

/**
 * @brief Store source data from device computations
 * @ingroup SourceDataAccess
 *
 * @tparam IndexType Point index type (non-SIMD)
 * @tparam PointSourceType Point source type matching medium and wavefield tags
 * @tparam DimensionTag Spatial dimension (deduced from sources parameter)
 * @param index Spectral element index identifying storage location
 * @param point_source [in] Source data to be stored in assembly
 * @param sources Source assembly with current timestep configuration
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 */
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag DimensionTag>
KOKKOS_INLINE_FUNCTION void store_on_device(
    const IndexType index, const PointSourceType &point_source,
    const specfem::assembly::sources<DimensionTag> &sources) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when storing sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag == DimensionTag,
                "PointSourceType dimension_tag must match sources dimension_tag");

  static_assert(IndexType::dimension_tag == DimensionTag,
                "IndexType dimension_tag must match sources dimension_tag");

#ifndef NDEBUG
  const int isource = index.imap;

  if ((index.ispec < 0) || (sources.nspec <= index.ispec)) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }
  if (sources.medium_types(isource) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }
  if (sources.wavefield_types(isource) != PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  using MediumTags = specfem::tags::Tags<PointSourceType::dimension_tag,
                                         PointSourceType::medium_tag>;

  sources.source_by_medium.template get<MediumTags>().store_on_device(
      sources.timestep, index, point_source);
}

/**
 * @brief Store source data from host computations
 * @ingroup SourceDataAccess
 *
 * @tparam IndexType Point index type (non-SIMD)
 * @tparam PointSourceType Point source type matching medium and wavefield tags
 * @tparam DimensionTag Spatial dimension (deduced from sources parameter)
 * @param index Spectral element index identifying storage location
 * @param point_source [in] Source data to be stored in assembly
 * @param sources Source assembly with current timestep configuration
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 */
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag DimensionTag>
void store_on_host(
    const IndexType index, const PointSourceType &point_source,
    const specfem::assembly::sources<DimensionTag> &sources) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when storing sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag == DimensionTag,
                "PointSourceType dimension_tag must match sources dimension_tag");

  static_assert(IndexType::dimension_tag == DimensionTag,
                "IndexType dimension_tag must match sources dimension_tag");

#ifndef NDEBUG
  const int isource = index.imap;

  if ((index.ispec < 0) || (sources.nspec <= index.ispec)) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }
  if (sources.h_medium_types(isource) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }
  if (sources.h_wavefield_types(isource) != PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  using MediumTags = specfem::tags::Tags<PointSourceType::dimension_tag,
                                         PointSourceType::medium_tag>;

  sources.source_by_medium.template get<MediumTags>().store_on_host(
      sources.timestep, index, point_source);
}

} // namespace specfem::assembly
