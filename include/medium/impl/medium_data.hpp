#pragma once

#include "domain_view.hpp"
#include "enumerations/medium.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include <cstddef>

#define DEFINE_MEDIUM_VIEW(prop, index_value)                                  \
  KOKKOS_INLINE_FUNCTION type_real prop(const int &ispec, const int &iz,       \
                                        const int &ix) const {                 \
    return base_type::data[index_value](ispec, iz, ix);                        \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION type_real h_##prop(const int &ispec, const int &iz,   \
                                            const int &ix) const {             \
    return base_type::h_data[index_value](ispec, iz, ix);                      \
  }

namespace specfem {
namespace medium {

namespace impl {

template <typename GlobalReducer, typename ContainerType, typename LocalReducer,
          typename PointValues, typename WorkItems>
void reduce(const ContainerType container, const WorkItems work_items,
            PointValues &values, const LocalReducer &reducer) {

  constexpr auto dimension = ContainerType::dimension;
  constexpr auto medium_tag = ContainerType::medium_tag;
  constexpr auto property_tag = ContainerType::property_tag;

  const int ngllz = container.ngllz;
  const int ngllx = container.ngllx;

  static_assert(PointValues::dimension == dimension, "Dimension mismatch");
  static_assert(PointValues::medium_tag == medium_tag, "Medium tag mismatch");
  static_assert(PointValues::property_tag == property_tag,
                "Property tag mismatch");

  constexpr std::size_t nprops = ContainerType::nprops;

  static_assert(nprops == PointValues::nprops,
                "Number of properties in PointValues must match the container");

  if (work_items.size() == 0)
    return;

  constexpr bool on_device = std::is_same<typename WorkItems::execution_space,
                                          Kokkos::DefaultExecutionSpace>::value;

  using policy_type = Kokkos::MDRangePolicy<
      typename WorkItems::execution_space,
      Kokkos::Rank<3, Kokkos::Iterate::Left> >; // Use the execution
                                                // space of the work
                                                // items

  const int nwork_items =
      work_items.extent(0); // Number of work items to be used for reduction

  for (std::size_t iprop = 0; iprop < nprops; ++iprop) {
    // Reduce the values in the container

    Kokkos::parallel_reduce(
        "reduce_medium_data",
        policy_type({ 0, 0, 0 }, { nwork_items, ngllz, ngllx }),
        KOKKOS_LAMBDA(const int i, const int iz, const int ix,
                      type_real &lvalue) {
          const int ispec = work_items(i); // Get the ispec from the work items
          // Access the data based on the device or host
          type_real value =
              container.template get_data<on_device>(ispec, iz, ix, i);
          // Perform Local reduction
          lvalue = reducer(value, lvalue);
        },
        GlobalReducer(values.data[iprop]));
  }

  return;
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, std::size_t N>
struct medium_data {
  using view_type = specfem::kokkos::DomainView2d<
      type_real, 3, Kokkos::DefaultExecutionSpace::memory_space>;
  constexpr static auto nprops = N;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;

  int nspec; ///< total number of acoustic spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int ngllx; ///< number of quadrature points in x dimension

  view_type data[N];
  typename view_type::HostMirror h_data[N];

  medium_data() = default;

  medium_data(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx) {

    for (std::size_t i = 0; i < nprops; i++) {
      data[i] = view_type("medium_data", nspec, ngllz, ngllx);
      h_data[i] = specfem::kokkos::create_mirror_view(data[i]);
    }
  }

private:
  template <bool on_device>
  KOKKOS_INLINE_FUNCTION type_real &get_data(const int &ispec, const int &iz,
                                             const int &ix,
                                             const std::size_t &i) const {
    if constexpr (on_device) {
      return data[i](ispec, iz, ix);
    } else {
      return h_data[i](ispec, iz, ix);
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

  template <typename WorkItems, typename PointValues>
  void max(const WorkItems &work_items, PointValues &values) const {

    for (std::size_t i = 0; i < values.nprops; ++i) {
      // Initialize the values to the minimum possible value
      // for max reduction
      values.data[i] = Kokkos::reduction_identity<type_real>::max();
    }
    // Reduce the values on device
    reduce<Kokkos::Max<type_real> >(
        *this, work_items, values,
        KOKKOS_LAMBDA(type_real value, type_real lvalue) {
          // Local reducer for max
          return (value > lvalue) ? value : lvalue;
        });

    return;
  }

  void copy_to_device() {
    for (std::size_t i = 0; i < nprops; i++) {
      Kokkos::deep_copy(data[i], h_data[i]);
    }
  }

  void copy_to_host() {
    for (std::size_t i = 0; i < nprops; i++) {
      Kokkos::deep_copy(h_data[i], data[i]);
    }
  }

  template <typename GlobalReducer, typename ContainerType,
            typename LocalReducer, typename PointValues, typename WorkItems>
  friend void reduce(const ContainerType container, const WorkItems work_items,
                     PointValues &values, const LocalReducer &reducer);
};
} // namespace impl

} // namespace medium
} // namespace specfem
