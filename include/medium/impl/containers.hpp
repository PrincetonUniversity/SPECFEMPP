#pragma once

#include "enumerations/medium.hpp"

#define DEFINE_MEDIUM_VIEW(prop)                                               \
  constexpr static int i_##prop = __COUNTER__ - _counter - 1;                  \
  KOKKOS_INLINE_FUNCTION type_real &prop(const int &ispec, const int &iz,      \
                                         const int &ix) const {                \
    return base_type::data(ispec, iz, ix, i_##prop);                           \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION type_real &h_##prop(const int &ispec, const int &iz,  \
                                             const int &ix) const {            \
    return base_type::h_data(ispec, iz, ix, i_##prop);                         \
  }

namespace specfem {
namespace medium {

namespace impl {
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int N>
struct medium_container {
  using view_type = typename Kokkos::View<type_real ***[N], Kokkos::LayoutLeft,
                                          Kokkos::DefaultExecutionSpace>;
  constexpr static auto nprops = N;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;

  int nspec; ///< total number of acoustic spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension

  view_type data;
  typename view_type::HostMirror h_data;

  medium_container() = default;

  medium_container(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        data("specfem::medium::impl::container::data", nspec, ngllz, ngllx, N),
        h_data(Kokkos::create_mirror_view(data)) {}

private:
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

    for (int i = 0; i < nprops; i++) {
      values.data[i] =
          on_device ? data(ispec, iz, ix, i) : h_data(ispec, iz, ix, i);
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

    for (int i = 0; i < nprops; i++) {
      Kokkos::Experimental::where(mask, values.data[i])
          .copy_from(on_device ? &data(ispec, iz, ix, i)
                               : &h_data(ispec, iz, ix, i),
                     tag_type());
    }
  }

  template <bool on_device, typename PointValues>
  inline void store_values(const specfem::point::index<dimension> &index,
                           const PointValues &values) const {

    static_assert(PointValues::dimension == dimension, "Dimension mismatch");
    static_assert(PointValues::medium_tag == medium_tag, "Medium tag mismatch");
    static_assert(PointValues::property_tag == property_tag,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    for (int i = 0; i < nprops; i++) {
      if constexpr (on_device) {
        data(ispec, iz, ix, i) = values.data[i];
      } else {
        h_data(ispec, iz, ix, i) = values.data[i];
      }
    }
  }

  template <bool on_device, typename PointValues>
  inline void store_values(const specfem::point::simd_index<dimension> &index,
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

    for (int i = 0; i < nprops; i++) {
      Kokkos::Experimental::where(mask, values.data[i])
          .copy_to(on_device ? &data(ispec, iz, ix, i)
                             : &h_data(ispec, iz, ix, i),
                   tag_type());
    }
  }

  template <bool on_device, typename PointValues>
  inline void add_values(const specfem::point::index<dimension> &index,
                         const PointValues &values) const {

    static_assert(PointValues::dimension == dimension, "Dimension mismatch");
    static_assert(PointValues::medium_tag == medium_tag, "Medium tag mismatch");
    static_assert(PointValues::property_tag == property_tag,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    for (int i = 0; i < nprops; i++) {
      if constexpr (on_device) {
        data(ispec, iz, ix, i) += values.data[i];
      } else {
        h_data(ispec, iz, ix, i) += values.data[i];
      }
    }
  }

  template <bool on_device, typename PointValues>
  inline void add_values(const specfem::point::simd_index<dimension> &index,
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

    for (int i = 0; i < nprops; i++) {
      Kokkos::Experimental::where(mask, lhs).copy_from(
          on_device ? &data(ispec, iz, ix, i) : &h_data(ispec, iz, ix, i),
          tag_type());
      lhs += values.data[i];
      Kokkos::Experimental::where(mask, lhs).copy_to(
          on_device ? &data(ispec, iz, ix, i) : &h_data(ispec, iz, ix, i),
          tag_type());
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
  store_device_values(const IndexType &index, PointValues &values) const {
    store_values<true>(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  store_host_values(const IndexType &index, PointValues &values) const {
    store_values<false>(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_device_values(const IndexType &index, PointValues &values) const {
    add_values<true>(index, values);
  }

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void add_host_values(const IndexType &index,
                                                   PointValues &values) const {
    add_values<false>(index, values);
  }

  void copy_to_device() { Kokkos::deep_copy(data, h_data); }

  void copy_to_host() { Kokkos::deep_copy(h_data, data); }
};
} // namespace impl

} // namespace medium
} // namespace specfem
