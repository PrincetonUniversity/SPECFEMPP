#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::boundary_values_impl {

template <specfem::element::medium_tag MediumTag,
          specfem::element::boundary_tag BoundaryTag>
class boundary_medium_container<specfem::dimension::type::dim3, MediumTag,
                                BoundaryTag> {
public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto boundary_tag = BoundaryTag;

private:
  constexpr static int components =
      specfem::element::attributes<dimension_tag, medium_tag>::components;

  using ValueViewType =
      Kokkos::View<type_real *****[components], Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
public:
  ///< store field values

  ValueViewType values;
  typename ValueViewType::HostMirror h_values;

  boundary_medium_container() = default;

  boundary_medium_container(const int nspec, const int ngllz, const int nglly,
                            const int ngllx, const int nstep)
      : values("specfem::assembly::impl::boundary_values", nspec, ngllz, nglly,
               ngllx, nstep),
        h_values(Kokkos::create_mirror_view(values)) {}

  boundary_medium_container(
      const int nstep, const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::assembly::boundaries<dimension_tag> boundaries,
      Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
          property_index_mapping);

  template <
      typename AccelerationType,
      typename std::enable_if_t<!AccelerationType::simd::using_simd, int> = 0>
  KOKKOS_FUNCTION void
  load_on_device(const int istep,
                 const specfem::point::index<dimension_tag> &index,
                 AccelerationType &acceleration) const {

    if (values.size() == 0)
      return;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int iy = index.iy;
    const int ix = index.ix;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
    for (int icomp = 0; icomp < components; ++icomp) {
      acceleration(icomp) = values(ispec, iz, iy, ix, istep, icomp);
    }

    return;
  }

  template <
      typename AccelerationType,
      typename std::enable_if_t<!AccelerationType::simd::using_simd, int> = 0>
  KOKKOS_FUNCTION void
  store_on_device(const int istep,
                  const specfem::point::index<dimension_tag> &index,
                  const AccelerationType &acceleration) const {

    if (values.size() == 0)
      return;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int iy = index.iy;
    const int ix = index.ix;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
    for (int icomp = 0; icomp < components; ++icomp) {
      values(ispec, iz, iy, ix, istep, icomp) = acceleration(icomp);
    }

    return;
  }

  template <
      typename AccelerationType,
      typename std::enable_if_t<AccelerationType::simd::using_simd, int> = 0>
  KOKKOS_FUNCTION void
  load_on_device(const int istep,
                 const specfem::point::simd_index<dimension_tag> &index,
                 AccelerationType &acceleration) const {

    if (values.size() == 0)
      return;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int iy = index.iy;
    const int ix = index.ix;

    using simd = typename AccelerationType::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int icomp = 0; icomp < components; ++icomp)
      Kokkos::Experimental::where(mask, acceleration(icomp))
          .copy_from(&values(ispec, iz, iy, ix, istep, icomp), tag_type());

    return;
  }

  template <
      typename AccelerationType,
      typename std::enable_if_t<AccelerationType::simd::using_simd, int> = 0>
  KOKKOS_FUNCTION void
  store_on_device(const int istep,
                  const specfem::point::simd_index<dimension_tag> &index,
                  const AccelerationType &acceleration) const {

    if (values.size() == 0)
      return;

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int iy = index.iy;
    const int ix = index.ix;

    using simd = typename AccelerationType::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int icomp = 0; icomp < components; ++icomp)
      Kokkos::Experimental::where(mask, acceleration(icomp))
          .copy_to(&values(ispec, iz, iy, ix, istep, icomp), tag_type());

    return;
  }

  void sync_to_host() {
    Kokkos::deep_copy(h_values, values);
    return;
  }

  void sync_to_device() {
    Kokkos::deep_copy(values, h_values);
    return;
  }
};

} // namespace specfem::assembly::boundary_values_impl
