#ifndef _COMPUTE_BOUNDARIES_VALUES_BOUNDARY_MEDIUM_CONTAINER_HPP
#define _COMPUTE_BOUNDARIES_VALUES_BOUNDARY_MEDIUM_CONTAINER_HPP

#include "compute/boundaries/boundaries.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
namespace impl {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::boundary_tag BoundaryTag>
class boundary_medium_container {
private:
  constexpr static int components =
      specfem::element::attributes<DimensionType, MediumTag>::components();
  constexpr static auto dimension = DimensionType;

public:
  using value_type =
      Kokkos::View<type_real ****[components], Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>;

  value_type values;
  typename value_type::HostMirror h_values;

  boundary_medium_container() = default;

  boundary_medium_container(const int nspec, const int nz, const int nx,
                            const int nstep)
      : values("specfem::compute::impl::stacey_values", nspec, nz, nx, nstep),
        h_values(Kokkos::create_mirror_view(values)) {}

  boundary_medium_container(
      const int nstep, const specfem::compute::mesh mesh,
      const specfem::compute::element_types element_types,
      const specfem::compute::boundaries boundaries,
      specfem::kokkos::HostView1d<int> property_index_mapping);

  template <
      typename AccelerationType,
      typename std::enable_if_t<!AccelerationType::simd::using_simd, int> = 0>
  KOKKOS_FUNCTION void
  load_on_device(const int istep, const specfem::point::index<dimension> &index,
                 AccelerationType &acceleration) const {

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int icomp = 0; icomp < components; ++icomp) {
      acceleration.acceleration(icomp) = values(ispec, iz, ix, istep, icomp);
    }

    return;
  }

  template <
      typename AccelerationType,
      typename std::enable_if_t<!AccelerationType::simd::using_simd, int> = 0>
  KOKKOS_FUNCTION void
  store_on_device(const int istep,
                  const specfem::point::index<dimension> &index,
                  const AccelerationType &acceleration) const {

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int icomp = 0; icomp < components; ++icomp) {
      values(ispec, iz, ix, istep, icomp) = acceleration.acceleration(icomp);
    }

    return;
  }

  template <
      typename AccelerationType,
      typename std::enable_if_t<AccelerationType::simd::using_simd, int> = 0>
  KOKKOS_FUNCTION void
  load_on_device(const int istep,
                 const specfem::point::simd_index<dimension> &index,
                 AccelerationType &acceleration) const {

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    using simd = typename AccelerationType::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int icomp = 0; icomp < components; ++icomp)
      Kokkos::Experimental::where(mask, acceleration.acceleration(icomp))
          .copy_from(&values(ispec, iz, ix, istep, icomp), tag_type());

    return;
  }

  template <
      typename AccelerationType,
      typename std::enable_if_t<AccelerationType::simd::using_simd, int> = 0>
  KOKKOS_FUNCTION void
  store_on_device(const int istep,
                  const specfem::point::simd_index<dimension> &index,
                  const AccelerationType &acceleration) const {

    const int ispec = index.ispec;
    const int iz = index.iz;
    const int ix = index.ix;

    using simd = typename AccelerationType::simd;
    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int icomp = 0; icomp < components; ++icomp)
      Kokkos::Experimental::where(mask, acceleration.acceleration(icomp))
          .copy_to(&values(ispec, iz, ix, istep, icomp), tag_type());

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

} // namespace impl
} // namespace compute
} // namespace specfem

#endif
