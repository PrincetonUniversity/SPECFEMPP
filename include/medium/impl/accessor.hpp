#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {
namespace impl {

template <typename DataContainer> class Accessor {

private:
  constexpr static auto dimension = specfem::dimension::type::dim2;

  template <typename PointValues>
  KOKKOS_INLINE_FUNCTION void
  get_data_on_device(const specfem::point::index<dimension> &index,
                     PointValues &values) const {
    static_cast<const DataContainer *>(this)->for_each_on_device_const(
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
  get_data_on_device(const specfem::point::simd_index<dimension> &index,
                     PointValues &values) const {

    using simd = typename PointValues::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });
    static_cast<const DataContainer *>(this)->for_each_on_device_const(
        index, [&](const type_real &value, const std::size_t i) mutable {
          Kokkos::Experimental::where(mask, values[i])
              .copy_from(&value, tag_type());
        });
  }

  template <typename PointValues>
  void get_data_on_host(const specfem::point::simd_index<dimension> &index,
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
  set_data_on_device(const specfem::point::simd_index<dimension> &index,
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
  void set_data_on_host(const specfem::point::simd_index<dimension> &index,
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
  add_data_on_device(const specfem::point::simd_index<dimension> &index,
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
  void add_data_on_host(const specfem::point::simd_index<dimension> &index,
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
  template <typename PointValues, typename IndexType>
  KOKKOS_INLINE_FUNCTION void load_device_values(const IndexType &index,
                                                 PointValues &values) const {
    get_data_on_device(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_INLINE_FUNCTION void load_host_values(const IndexType &index,
                                               PointValues &values) const {
    get_data_on_host(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_INLINE_FUNCTION void
  store_device_values(const IndexType &index, const PointValues &values) const {
    set_data_on_device(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_INLINE_FUNCTION void
  store_host_values(const IndexType &index, const PointValues &values) const {
    set_data_on_host(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_INLINE_FUNCTION void
  add_device_values(const IndexType &index, const PointValues &values) const {

    add_data_on_device(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_INLINE_FUNCTION void add_host_values(const IndexType &index,
                                              const PointValues &values) const {
    add_data_on_host(index, values);
  }
};

} // namespace impl
} // namespace medium
} // namespace specfem
