#pragma once

#include "enumerations/medium.hpp"

#define DEFINE_CONTAINER(prop)                                                 \
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
struct impl_properties_container {
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

  impl_properties_container() = default;

  impl_properties_container(const int nspec, const int ngllz, const int ngllx)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
        data("specfem::benchmarks::properties::data", nspec, ngllz, ngllx, N),
        h_data(Kokkos::create_mirror_view(data)) {}

private:
  template <bool on_device, typename PointProperties>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_properties(const specfem::point::index<dimension> &index,
                  PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == medium_tag,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_tag,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    for (int i = 0; i < nprops; i++) {
      property.data[i] =
          on_device ? data(ispec, iz, ix, i) : h_data(ispec, iz, ix, i);
    }
  }

  template <bool on_device, typename PointProperties>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_properties(const specfem::point::simd_index<dimension> &index,
                  PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == medium_tag,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_tag,
                  "Property tag mismatch");

    using simd = typename PointProperties::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    mask_type mask([&, this](std::size_t lane) { return index.mask(lane); });

    for (int i = 0; i < nprops; i++) {
      Kokkos::Experimental::where(mask, property.data[i])
          .copy_from(on_device ? &data(ispec, iz, ix, i)
                               : &h_data(ispec, iz, ix, i),
                     tag_type());
    }
  }

public:
  template <typename PointProperties, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_device_properties(const IndexType &index,
                         PointProperties &property) const {
    load_properties<true>(index, property);
  }

  template <typename PointProperties, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_host_properties(const IndexType &index,
                       PointProperties &property) const {
    load_properties<false>(index, property);
  }

  template <typename PointProperties>
  inline void assign(const specfem::point::index<dimension> &index,
                     const PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == medium_tag,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_tag,
                  "Property tag mismatch");

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    for (int i = 0; i < nprops; i++) {
      h_data(ispec, iz, ix, i) = property.data[i];
    }
  }

  template <typename PointProperties>
  inline void assign(const specfem::point::simd_index<dimension> &index,
                     const PointProperties &property) const {

    static_assert(PointProperties::dimension == dimension,
                  "Dimension mismatch");
    static_assert(PointProperties::medium_tag == medium_tag,
                  "Medium tag mismatch");
    static_assert(PointProperties::property_tag == property_tag,
                  "Property tag mismatch");

    using simd = typename PointProperties::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    mask_type mask([&, this](std::size_t lane) { return index.mask(lane); });

    for (int i = 0; i < nprops; i++) {
      Kokkos::Experimental::where(mask, property.data[i])
          .copy_to(&h_data(ispec, iz, ix, i), tag_type());
    }
  }

  void copy_to_device() { Kokkos::deep_copy(data, h_data); }

  void copy_to_host() { Kokkos::deep_copy(h_data, data); }
};
} // namespace impl

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
struct properties_container {
  static_assert("Material type not implemented");
};

} // namespace medium
} // namespace specfem

// Including the template specializations here so that properties_container is
// an interface to the compute module
#include "dim2/acoustic/isotropic/properties_container.hpp"
#include "dim2/elastic/anisotropic/properties_container.hpp"
#include "dim2/elastic/isotropic/properties_container.hpp"

#undef DEFINE_CONTAINER
