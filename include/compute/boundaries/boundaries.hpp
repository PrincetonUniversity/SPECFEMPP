#ifndef _COMPUTE_BOUNDARIES_BOUNDARIES_HPP
#define _COMPUTE_BOUNDARIES_BOUNDARIES_HPP

#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/specfem_enums.hpp"
#include "macros.hpp"
#include "point/boundary.hpp"

namespace specfem {
namespace compute {
struct boundaries {

  boundaries() = default;

  boundaries(const int nspec, const int ngllz, const int ngllx,
             const specfem::compute::mesh_to_compute_mapping &mapping,
             const specfem::mesh::tags &tags,
             const specfem::compute::properties &properties,
             const specfem::mesh::boundaries &boundary);

  specfem::kokkos::DeviceView1d<specfem::element::boundary_tag_container>
      boundary_tags; ///< Boundary tags for each element
  specfem::kokkos::HostMirror1d<specfem::element::boundary_tag_container>
      h_boundary_tags; ///< Host mirror of boundary tags

  specfem::kokkos::DeviceView3d<specfem::element::boundary_tag_container>
      boundary_types; ///< Boundary types for each element
  specfem::kokkos::HostMirror3d<specfem::element::boundary_tag_container>
      h_boundary_types; ///< Host mirror of boundary types

  //   specfem::compute::boundaries::impl::stacey_values<specfem::element::type::acoustic>
  //       stacey; ///< Stacey boundary

  //   specfem::compute::impl::boundaries::boundary_container<
  //       specfem::enums::element::boundary_tag::acoustic_free_surface>
  //       acoustic_free_surface; ///< Acoustic free surface boundary

  //   specfem::compute::impl::boundaries::boundary_container<
  //       specfem::enums::element::boundary_tag::stacey>
  //       stacey; ///< Stacey boundary

  //   specfem::compute::impl::boundaries::boundary_container<
  //       specfem::enums::element::boundary_tag::composite_stacey_dirichlet>
  //       composite_stacey_dirichlet; ///< Composite Stacey and Dirichlet
  //       boundary
};

template <typename PointBoundaryType,
          typename std::enable_if_t<PointBoundaryType::isPointBoundaryType &&
                                        !PointBoundaryType::simd::using_simd,
                                    int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const specfem::point::index &index,
               const specfem::compute::boundaries &boundaries,
               PointBoundaryType &boundary) {

  constexpr auto tag = PointBoundaryType::boundary_tag;

  if constexpr (tag == specfem::element::boundary_tag::none)
    return;

  boundary.tags[0] = boundaries.boundary_types(index.ispec, index.iz, index.ix);
  return;
}

template <typename PointBoundaryType,
          typename std::enable_if_t<PointBoundaryType::isPointBoundaryType &&
                                        PointBoundaryType::simd::using_simd,
                                    int> = 0>
NOINLINE KOKKOS_FUNCTION void
load_on_device(const specfem::point::simd_index &index,
               const specfem::compute::boundaries &boundaries,
               PointBoundaryType &boundary) {

  constexpr auto tag = PointBoundaryType::boundary_tag;

  if constexpr (tag == specfem::element::boundary_tag::none)
    return;

  constexpr int simd_size = PointBoundaryType::simd::size();

  for (int lane = 0; lane < simd_size; ++lane) {
    if (index.mask(lane))
      boundary.tags[lane] =
          boundaries.boundary_types(index.ispec + lane, index.iz, index.ix);
  }

  return;
}

template <typename PointBoundaryType,
          typename std::enable_if_t<PointBoundaryType::isPointBoundaryType &&
                                        !PointBoundaryType::simd::using_simd,
                                    int> = 0>
void load_on_host(const specfem::point::index &index,
                  const specfem::compute::boundaries &boundaries,
                  PointBoundaryType &boundary) {

  constexpr auto tag = PointBoundaryType::boundary_tag;

  if constexpr (tag == specfem::element::boundary_tag::none)
    return;

  boundary.tags[0] =
      boundaries.h_boundary_types(index.ispec, index.iz, index.ix);
  return;
}

template <typename PointBoundaryType,
          typename std::enable_if_t<PointBoundaryType::isPointBoundaryType &&
                                        PointBoundaryType::simd::using_simd,
                                    int> = 0>
void load_on_host(const specfem::point::simd_index &index,
                  const specfem::compute::boundaries &boundaries,
                  PointBoundaryType &boundary) {

  constexpr auto tag = PointBoundaryType::boundary_tag;

  if constexpr (tag == specfem::element::boundary_tag::none)
    return;

  constexpr int simd_size = PointBoundaryType::simd::size();

  for (int lane = 0; lane < simd_size; ++lane) {
    if (index.mask(lane))
      boundary.tags[lane] =
          boundaries.h_boundary_types(index.ispec + lane, index.iz, index.ix);
  }
  return;
}

} // namespace compute
} // namespace specfem

#endif
