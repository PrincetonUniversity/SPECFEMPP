#pragma once

#include "domain_view.hpp"
#include "enumerations/medium.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include <cstddef>

#define DEFINE_MEDIUM_VIEW(prop, index_value)                                  \
  KOKKOS_INLINE_FUNCTION type_real prop(const int &ispec, const int &iz,       \
                                        const int &ix) const {                 \
    return base_type::data(ispec, iz, ix, index_value);                        \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION type_real h_##prop(const int &ispec, const int &iz,   \
                                            const int &ix) const {             \
    return base_type::h_data(ispec, iz, ix, index_value);                      \
  }

namespace specfem {
namespace medium {

namespace impl {
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, std::size_t N>
struct medium_data {
  using view_type = Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                 Kokkos::DefaultExecutionSpace::memory_space>;
  constexpr static auto nprops = N;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;

  int nspec; ///< total number of acoustic spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension

  view_type data;
  typename view_type::HostMirror h_data;

  medium_data() = default;

  medium_data(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        data("specfem::medium::impl::container::data", nspec, ngllz, ngllx, N),
        h_data(Kokkos::create_mirror_view(data)) {}

private:
  template <bool on_device>
  KOKKOS_INLINE_FUNCTION type_real &get_data(const int &ispec, const int &iz,
                                             const int &ix,
                                             const std::size_t &i) const {
    if constexpr (on_device) {
      return data(ispec, iz, ix, i);
    } else {
      return h_data(ispec, iz, ix, i);
    }
  }

  template <bool on_device, typename PointValues>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_values(const specfem::point::index<dimension> &index,
              PointValues &values) const {

    static_assert(PointValues::dimension == dimension, "Dimension mismatch");
    static_assert(PointValues::medium_tag == medium_tag, "Medium tag mismatch");
    static_assert(PointValues::property_tag == property_tag,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    /// std::size_t is required to avoid intel compiler optimization issue
    /// Intel compiler optimizes this loop incorrectly when the loop is on int
    /// This loop is a compile time pragma unroll
    for (std::size_t i = 0; i < nprops; i++) {
      values.data[i] = get_data<on_device>(ispec, iz, ix, i);
    }
  }

  template <bool on_device, typename PointValues>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_values(const specfem::point::simd_index<dimension> &index,
              PointValues &values) const {

    static_assert(PointValues::dimension == dimension, "Dimension mismatch");
    static_assert(PointValues::medium_tag == medium_tag, "Medium tag mismatch");
    static_assert(PointValues::property_tag == property_tag,
                  "Property tag mismatch");

    using simd = typename PointValues::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    mask_type mask([&, this](std::size_t lane) { return index.mask(lane); });

    /// std::size_t is required to avoid intel compiler optimization issue
    /// Intel compiler optimizes this loop incorrectly when the loop is on int
    /// This loop is a compile time pragma unroll
    for (std::size_t i = 0; i < nprops; i++) {
      Kokkos::Experimental::where(mask, values.data[i])
          .copy_from(&get_data<on_device>(ispec, iz, ix, i), tag_type());
    }
  }

  template <bool on_device, typename PointValues>
  KOKKOS_FORCEINLINE_FUNCTION void
  store_values(const specfem::point::index<dimension> &index,
               const PointValues &values) const {

    static_assert(PointValues::dimension == dimension, "Dimension mismatch");
    static_assert(PointValues::medium_tag == medium_tag, "Medium tag mismatch");
    static_assert(PointValues::property_tag == property_tag,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    /// std::size_t is required to avoid intel compiler optimization issue
    /// Intel compiler optimizes this loop incorrectly when the loop is on int
    /// This loop is a compile time pragma unroll
    for (std::size_t i = 0; i < nprops; i++) {
      get_data<on_device>(ispec, iz, ix, i) = values.data[i];
    }
  }

  template <bool on_device, typename PointValues>
  KOKKOS_FORCEINLINE_FUNCTION void
  store_values(const specfem::point::simd_index<dimension> &index,
               const PointValues &values) const {

    static_assert(PointValues::dimension == dimension, "Dimension mismatch");
    static_assert(PointValues::medium_tag == medium_tag, "Medium tag mismatch");
    static_assert(PointValues::property_tag == property_tag,
                  "Property tag mismatch");

    using simd = typename PointValues::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    mask_type mask([&, this](std::size_t lane) { return index.mask(lane); });

    /// std::size_t is required to avoid intel compiler optimization issue
    /// Intel compiler optimizes this loop incorrectly when the loop is on int
    /// This loop is a compile time pragma unroll
    for (std::size_t i = 0; i < nprops; i++) {
      Kokkos::Experimental::where(mask, values.data[i])
          .copy_to(&get_data<on_device>(ispec, iz, ix, i), tag_type());
    }
  }

  template <bool on_device, typename PointValues>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_values(const specfem::point::index<dimension> &index,
             const PointValues &values) const {

    static_assert(PointValues::dimension == dimension, "Dimension mismatch");
    static_assert(PointValues::medium_tag == medium_tag, "Medium tag mismatch");
    static_assert(PointValues::property_tag == property_tag,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    /// std::size_t is required to avoid intel compiler optimization issue
    /// Intel compiler optimizes this loop incorrectly when the loop is on int
    /// This loop is a compile time pragma unroll
    for (std::size_t i = 0; i < nprops; i++) {
      get_data<on_device>(ispec, iz, ix, i) += values.data[i];
    }
  }

  template <bool on_device, typename PointValues>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_values(const specfem::point::simd_index<dimension> &index,
             const PointValues &values) const {

    static_assert(PointValues::dimension == dimension, "Dimension mismatch");
    static_assert(PointValues::medium_tag == medium_tag, "Medium tag mismatch");
    static_assert(PointValues::property_tag == property_tag,
                  "Property tag mismatch");

    using simd = typename PointValues::simd;
    using simd_type = typename simd::datatype;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    simd_type lhs;

    mask_type mask([&, this](std::size_t lane) { return index.mask(lane); });

    /// std::size_t is required to avoid intel compiler optimization issue
    /// Intel compiler optimizes this loop incorrectly when the loop is on int
    /// This loop is a compile time pragma unroll
    for (std::size_t i = 0; i < nprops; i++) {
      Kokkos::Experimental::where(mask, lhs).copy_from(
          &get_data<on_device>(ispec, iz, ix, i), tag_type());
      lhs += values.data[i];
      Kokkos::Experimental::where(mask, lhs).copy_to(
          &get_data<on_device>(ispec, iz, ix, i), tag_type());
    }
  }

public:
  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_device_values(const IndexType &index, PointValues &values) const {
    load_values<true>(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void load_host_values(const IndexType &index,
                                                    PointValues &values) const {
    load_values<false>(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  store_device_values(const IndexType &index, const PointValues &values) const {
    store_values<true>(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  store_host_values(const IndexType &index, const PointValues &values) const {
    store_values<false>(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_device_values(const IndexType &index, const PointValues &values) const {
    add_values<true>(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_host_values(const IndexType &index, const PointValues &values) const {
    add_values<false>(index, values);
  }

  void copy_to_device() { Kokkos::deep_copy(data, h_data); }

  void copy_to_host() { Kokkos::deep_copy(h_data, data); }
};
} // namespace impl

} // namespace medium
} // namespace specfem
