#pragma once

#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::impl {

/**
 * @brief CRTP base for accessing data containers in device/host memory.
 *
 * Provides unified interface for loading, storing, and accumulating point data
 * from containers at specific quadrature point indices. Uses CRTP to delegate
 * actual data access to derived container classes while providing common
 * interface patterns.
 *
 * Supports both scalar and SIMD vectorized operations, with automatic
 * dispatch based on index type. All device functions are Kokkos-compatible.
 *
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 * @tparam DataContainer Container type inheriting from this accessor
 *
 */
template <specfem::element::dimension_tag DimensionTag, typename DataContainer>
class DomainAccessor {

private:
  constexpr static auto dimension = DimensionTag;

  template <typename PointValues>
  KOKKOS_INLINE_FUNCTION void
  get_data_on_device(const specfem::point::index<dimension> &index,
                     PointValues &values) const {
    static_cast<const DataContainer *>(this)->for_each_on_device(
        index, [&](const type_real &value, const std::size_t i) mutable {
          values[i] = value;
        });
  }

  template <typename PointValues>
  void get_data_on_host(const specfem::point::index<dimension> &index,
                        PointValues &values) const {
    static_cast<const DataContainer *>(this)->for_each_on_host(
        index, [&](const type_real &value, const std::size_t i) mutable {
          values[i] = value;
        });
  }

  template <typename PointValues>
  KOKKOS_INLINE_FUNCTION void
  get_data_on_device(const specfem::point::index<dimension, true> &index,
                     PointValues &values) const {

    using simd = typename PointValues::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });
    static_cast<const DataContainer *>(this)->for_each_on_device(
        index, [&](const type_real &value, const std::size_t i) mutable {
          Kokkos::Experimental::where(mask, values[i])
              .copy_from(&value, tag_type());
        });
  }

  template <typename PointValues>
  void get_data_on_host(const specfem::point::index<dimension, true> &index,
                        PointValues &values) const {

    using simd = typename PointValues::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });
    static_cast<const DataContainer *>(this)->for_each_on_host(
        index, [&](const type_real &value, const std::size_t i) mutable {
          Kokkos::Experimental::where(mask, values[i])
              .copy_from(&value, tag_type());
        });
  }

  template <typename PointValues>
  KOKKOS_INLINE_FUNCTION void
  set_data_on_device(const specfem::point::index<dimension> &index,
                     const PointValues &values) const {
    static_cast<const DataContainer *>(this)->for_each_on_device(
        index,
        [&](type_real &value, const std::size_t i) { value = values[i]; });
  }

  template <typename PointValues>
  void set_data_on_host(const specfem::point::index<dimension> &index,
                        const PointValues &values) const {
    static_cast<const DataContainer *>(this)->for_each_on_host(
        index,
        [&](type_real &value, const std::size_t i) { value = values[i]; });
  }

  template <typename PointValues>
  KOKKOS_INLINE_FUNCTION void
  set_data_on_device(const specfem::point::index<dimension, true> &index,
                     const PointValues &values) const {

    using simd = typename PointValues::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });
    static_cast<const DataContainer *>(this)->for_each_on_device(
        index, [&](type_real &value, const std::size_t i) {
          Kokkos::Experimental::where(mask, values[i])
              .copy_to(&value, tag_type());
        });
  }

  template <typename PointValues>
  void set_data_on_host(const specfem::point::index<dimension, true> &index,
                        const PointValues &values) const {

    using simd = typename PointValues::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });
    static_cast<const DataContainer *>(this)->for_each_on_host(
        index, [&](type_real &value, const std::size_t i) {
          Kokkos::Experimental::where(mask, values[i])
              .copy_to(&value, tag_type());
        });
  }

  template <typename PointValues>
  KOKKOS_INLINE_FUNCTION void
  add_data_on_device(const specfem::point::index<dimension> &index,
                     const PointValues &values) const {
    static_cast<const DataContainer *>(this)->for_each_on_device(
        index,
        [&](type_real &value, const std::size_t i) { value += values[i]; });
  }

  template <typename PointValues>
  void add_data_on_host(const specfem::point::index<dimension> &index,
                        const PointValues &values) const {
    static_cast<const DataContainer *>(this)->for_each_on_host(
        index,
        [&](type_real &value, const std::size_t i) { value += values[i]; });
  }

  template <typename PointValues>
  KOKKOS_INLINE_FUNCTION void
  add_data_on_device(const specfem::point::index<dimension, true> &index,
                     const PointValues &values) const {

    using simd = typename PointValues::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&, this](std::size_t lane) { return index.mask(lane); });
    static_cast<const DataContainer *>(this)->for_each_on_device(
        index, [&](type_real &value, const std::size_t i) {
          typename PointValues::value_type temp;
          Kokkos::Experimental::where(mask, temp).copy_from(&value, tag_type());
          temp += values[i];
          Kokkos::Experimental::where(mask, temp).copy_to(&value, tag_type());
        });
  }

  template <typename PointValues>
  void add_data_on_host(const specfem::point::index<dimension, true> &index,
                        const PointValues &values) const {

    using simd = typename PointValues::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&, this](std::size_t lane) { return index.mask(lane); });
    static_cast<const DataContainer *>(this)->for_each_on_host(
        index, [&](type_real &value, const std::size_t i) {
          typename PointValues::value_type temp;
          Kokkos::Experimental::where(mask, temp).copy_from(&value, tag_type());
          temp += values[i];
          Kokkos::Experimental::where(mask, temp).copy_to(&value, tag_type());
        });
  }

public:
  /**
   * @brief Load data from container to point values on device.
   *
   * Reads data from the container at the specified index and copies it
   * to the provided point values object. Supports both scalar and SIMD
   * vectorized indices for optimal performance.
   *
   * @tparam PointValues Point data type (properties, field, etc.)
   * @tparam IndexType Index type (index or simd_index)
   * @param index Quadrature point index within element
   * @param values Point values object to fill with data
   */
  template <typename PointValues, typename IndexType>
  KOKKOS_INLINE_FUNCTION void load_device_values(const IndexType &index,
                                                 PointValues &values) const {
    get_data_on_device(index, values);
  }

  /**
   * @brief Load data from container to point values on host.
   *
   * Host version of load_device_values. Should be called from host code
   * for accessing data in host memory or performing host-side operations.
   *
   * @tparam PointValues Point data type (properties, field, etc.)
   * @tparam IndexType Index type (index or simd_index)
   * @param index Quadrature point index within element
   * @param values Point values object to fill with data
   */
  template <typename PointValues, typename IndexType>
  void load_host_values(const IndexType &index, PointValues &values) const {
    get_data_on_host(index, values);
  }

  /**
   * @brief Store point values to container on device.
   *
   * Writes data from point values object to the container at the specified
   * index. Overwrites existing data in the container.
   *
   * @tparam PointValues Point data type (properties, field, etc.)
   * @tparam IndexType Index type (index or simd_index)
   * @param index Quadrature point index within element
   * @param values Point values object containing data to store
   */
  template <typename PointValues, typename IndexType>
  KOKKOS_INLINE_FUNCTION void
  store_device_values(const IndexType &index, const PointValues &values) const {
    set_data_on_device(index, values);
  }

  /**
   * @brief Store point values to container on host.
   *
   * Host version of store_device_values for host-side operations.
   *
   * @tparam PointValues Point data type (properties, field, etc.)
   * @tparam IndexType Index type (index or simd_index)
   * @param index Quadrature point index within element
   * @param values Point values object containing data to store
   */
  template <typename PointValues, typename IndexType>
  void store_host_values(const IndexType &index,
                         const PointValues &values) const {
    set_data_on_host(index, values);
  }

  /**
   * @brief Add point values to existing container data on device.
   *
   * Performs element-wise addition of point values to existing data
   * in the container. Commonly used for accumulating contributions
   * (e.g., sensitivity kernels, forces).
   *
   * @tparam PointValues Point data type (properties, field, etc.)
   * @tparam IndexType Index type (index or simd_index)
   * @param index Quadrature point index within element
   * @param values Point values to add to existing container data
   */
  template <typename PointValues, typename IndexType>
  KOKKOS_INLINE_FUNCTION void
  add_device_values(const IndexType &index, const PointValues &values) const {

    add_data_on_device(index, values);
  }

  /**
   * @brief Add point values to existing container data on host.
   *
   * Host version of add_device_values for accumulation operations.
   *
   * @tparam PointValues Point data type (properties, field, etc.)
   * @tparam IndexType Index type (index or simd_index)
   * @param index Quadrature point index within element
   * @param values Point values to add to existing container data
   */
  template <typename PointValues, typename IndexType>
  void add_host_values(const IndexType &index,
                       const PointValues &values) const {
    add_data_on_host(index, values);
  }
};

} // namespace specfem::assembly::impl
