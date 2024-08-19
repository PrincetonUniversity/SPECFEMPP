#ifndef _COMPUTE_BOUNDARIES_BOUNDARIES_HPP
#define _COMPUTE_BOUNDARIES_BOUNDARIES_HPP

#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/specfem_enums.hpp"
#include "impl/acoustic_free_surface.hpp"
#include "impl/stacey.hpp"
#include "macros.hpp"
#include "point/boundary.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace compute {
struct boundaries {

  boundaries() = default;

  specfem::kokkos::HostView1d<specfem::element::boundary_tag> boundary_tags;

  specfem::kokkos::DeviceView1d<int> acoustic_free_surface_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_acoustic_free_surface_index_mapping;

  specfem::kokkos::DeviceView1d<int> stacey_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_stacey_index_mapping;

  specfem::compute::impl::boundaries::acoustic_free_surface
      acoustic_free_surface; ///< Acoustic free surface boundary

  specfem::compute::impl::boundaries::stacey stacey; ///< Stacey boundary

  boundaries(const int nspec, const int ngllz, const int ngllx,
             const specfem::mesh::mesh &mesh,
             const specfem::compute::mesh_to_compute_mapping &mapping,
             const specfem::compute::quadrature &quadrature,
             const specfem::compute::properties &properties,
             const specfem::compute::partial_derivatives &partial_derivatives);
};

template <typename IndexType, typename PointBoundaryType>
KOKKOS_INLINE_FUNCTION void
load_on_device(const IndexType &index,
               const specfem::compute::boundaries &boundaries,
               PointBoundaryType &boundary) {

  constexpr auto tag = PointBoundaryType::boundary_tag;

  static_assert(
      (tag == specfem::element::boundary_tag::none ||
       tag == specfem::element::boundary_tag::acoustic_free_surface ||
       tag == specfem::element::boundary_tag::stacey ||
       tag == specfem::element::boundary_tag::composite_stacey_dirichlet),
      "Boundary tag must be acoustic free surface, stacey, or "
      "composite_stacey_dirichlet");

  if constexpr (tag == specfem::element::boundary_tag::none)
    return;

  IndexType l_index = index;

  if constexpr (tag == specfem::element::boundary_tag::acoustic_free_surface) {
    l_index.ispec = boundaries.acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_device(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::stacey) {
    l_index.ispec = boundaries.stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_device(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::
                                  composite_stacey_dirichlet) {
    l_index.ispec = boundaries.acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_device(l_index, boundary);
    l_index.ispec = boundaries.stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_device(l_index, boundary);
  }

  return;
}

template <typename IndexType, typename PointBoundaryType>
inline void load_on_host(const IndexType &index,
                         const specfem::compute::boundaries &boundaries,
                         PointBoundaryType &boundary) {

  constexpr auto tag = PointBoundaryType::boundary_tag;

  static_assert(
      (tag == specfem::element::boundary_tag::none ||
       tag == specfem::element::boundary_tag::acoustic_free_surface ||
       tag == specfem::element::boundary_tag::stacey ||
       tag == specfem::element::boundary_tag::composite_stacey_dirichlet),
      "Boundary tag must be acoustic free surface, stacey, or "
      "composite_stacey_dirichlet");

  if constexpr (tag == specfem::element::boundary_tag::none)
    return;

  IndexType l_index = index;

  if constexpr (tag == specfem::element::boundary_tag::acoustic_free_surface) {
    l_index.ispec =
        boundaries.h_acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_host(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::stacey) {
    l_index.ispec = boundaries.h_stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_host(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::
                                  composite_stacey_dirichlet) {
    l_index.ispec =
        boundaries.h_acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_host(l_index, boundary);
    l_index.ispec = boundaries.h_stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_host(l_index, boundary);
  }

  return;
}

} // namespace compute
} // namespace specfem

#endif
