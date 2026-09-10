#pragma once

#include "impl/acoustic_free_surface.hpp"
#include "impl/stacey.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {
/**
 * @brief Boundary condition information for every quadrature point in a 3D
 * finite element mesh
 *
 * Stores per-element boundary tags and per-quadrature-point data for
 * acoustic free surface and Stacey absorbing boundary conditions in 3D.
 */
template <> struct boundaries<specfem::element::dimension_tag::dim3> {

private:
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using BoundaryViewType = Kokkos::View<specfem::element::boundary_tag *,
                                        Kokkos::DefaultExecutionSpace>;

public:
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag

  BoundaryViewType boundary_tags; ///< Per-element boundary tags (device)
  BoundaryViewType::host_mirror_type h_boundary_tags; ///< Host mirror

  IndexViewType acoustic_free_surface_index_mapping; ///< Compute→local index
                                                     ///< for
                                                     ///< acoustic_free_surface
  IndexViewType::host_mirror_type
      h_acoustic_free_surface_index_mapping; ///< Host mirror

  IndexViewType stacey_index_mapping; ///< Compute→local index for stacey
  IndexViewType::host_mirror_type h_stacey_index_mapping; ///< Host mirror

  specfem::assembly::boundaries_impl::acoustic_free_surface<dimension_tag>
      acoustic_free_surface; ///< Acoustic free surface BC data

  specfem::assembly::boundaries_impl::stacey<dimension_tag>
      stacey; ///< Stacey
              ///< absorbing
              ///< BC data

  /**
   * @name Constructors
   */
  ///@{
  boundaries() = default;

  /**
   * @brief Construct boundary condition data from mesh information
   *
   * @param nspec Number of spectral elements
   * @param ngllz GLL points in z direction
   * @param nglly GLL points in y direction
   * @param ngllx GLL points in x direction
   * @param mesh Finite element mesh with boundary information
   * @param mesh_assembly Assembly mesh with coordinate and mapping info
   * @param jacobian_matrix Jacobian matrix at every quadrature point
   */
  template <specfem::simulation::model ModelTag>
  boundaries(
      const int nspec, const int ngllz, const int nglly, const int ngllx,
      const specfem::mesh::mesh<ModelTag> &mesh,
      const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix);
  ///@}
};

/**
 * @defgroup BoundaryConditionDataAccess
 */

/**
 * @brief Load boundary condition information for a quadrature point on device
 *
 * @ingroup BoundaryConditionDataAccess
 *
 * @tparam IndexType Index type (specfem::point::index for dim3)
 * @tparam PointBoundaryType Point boundary type (specfem::point::boundary)
 * @param index Quadrature point index
 * @param boundaries Assembly boundary data
 * @param boundary Output boundary data for the quadrature point
 */
template <typename IndexType, typename PointBoundaryType,
          typename std::enable_if<PointBoundaryType::simd::using_simd ==
                                      IndexType::using_simd,
                                  int>::type = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
    const IndexType &index,
    const specfem::assembly::boundaries<specfem::element::dimension_tag::dim3>
        &boundaries,
    PointBoundaryType &boundary) {

  constexpr auto tag = PointBoundaryType::boundary_tag;

  static_assert(
      (tag == specfem::element::boundary_tag::none ||
       tag == specfem::element::boundary_tag::acoustic_free_surface ||
       tag == specfem::element::boundary_tag::stacey ||
       tag == specfem::element::boundary_tag::composite_stacey_dirichlet),
      "Boundary tag must be none, acoustic_free_surface, stacey, or "
      "composite_stacey_dirichlet");

  if constexpr (tag == specfem::element::boundary_tag::none)
    return;

  IndexType l_index = index;

  if constexpr (tag == specfem::element::boundary_tag::acoustic_free_surface) {
#ifndef NDEBUG
    if (boundaries.boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::acoustic_free_surface) {
      Kokkos::abort("Boundary tag for acoustic free surface does not match");
    }
#endif
    l_index.ispec = boundaries.acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_device(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::stacey) {
#ifndef NDEBUG
    if (boundaries.boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::stacey) {
      Kokkos::abort("Boundary tag for stacey does not match");
    }
#endif
    l_index.ispec = boundaries.stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_device(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::
                                  composite_stacey_dirichlet) {
#ifndef NDEBUG
    if (boundaries.boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::composite_stacey_dirichlet) {
      Kokkos::abort(
          "Boundary tag for composite_stacey_dirichlet does not match");
    }
#endif
    l_index.ispec = boundaries.acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_device(l_index, boundary);
    l_index.ispec = boundaries.stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_device(l_index, boundary);
  }

  return;
}

/**
 * @brief Load boundary condition information for a quadrature point on host
 *
 * @ingroup BoundaryConditionDataAccess
 *
 * @tparam IndexType Index type (specfem::point::index for dim3)
 * @tparam PointBoundaryType Point boundary type (specfem::point::boundary)
 * @param index Quadrature point index
 * @param boundaries Assembly boundary data
 * @param boundary Output boundary data for the quadrature point
 */
template <typename IndexType, typename PointBoundaryType,
          typename std::enable_if<PointBoundaryType::simd::using_simd ==
                                      IndexType::using_simd,
                                  int>::type = 0>
inline void load_on_host(
    const IndexType &index,
    const specfem::assembly::boundaries<specfem::element::dimension_tag::dim3>
        &boundaries,
    PointBoundaryType &boundary) {

  constexpr auto tag = PointBoundaryType::boundary_tag;

  static_assert(
      (tag == specfem::element::boundary_tag::none ||
       tag == specfem::element::boundary_tag::acoustic_free_surface ||
       tag == specfem::element::boundary_tag::stacey ||
       tag == specfem::element::boundary_tag::composite_stacey_dirichlet),
      "Boundary tag must be none, acoustic_free_surface, stacey, or "
      "composite_stacey_dirichlet");

  if constexpr (tag == specfem::element::boundary_tag::none)
    return;

  IndexType l_index = index;

  if constexpr (tag == specfem::element::boundary_tag::acoustic_free_surface) {
#ifndef NDEBUG
    if (boundaries.h_boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::acoustic_free_surface) {
      Kokkos::abort("Boundary tag for acoustic free surface does not match");
    }
#endif
    l_index.ispec =
        boundaries.h_acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_host(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::stacey) {
#ifndef NDEBUG
    if (boundaries.h_boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::stacey) {
      Kokkos::abort("Boundary tag for stacey does not match");
    }
#endif
    l_index.ispec = boundaries.h_stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_host(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::
                                  composite_stacey_dirichlet) {
#ifndef NDEBUG
    if (boundaries.h_boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::composite_stacey_dirichlet) {
      Kokkos::abort(
          "Boundary tag for composite_stacey_dirichlet does not match");
    }
#endif
    l_index.ispec =
        boundaries.h_acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_host(l_index, boundary);
    l_index.ispec = boundaries.h_stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_host(l_index, boundary);
  }

  return;
}

} // namespace specfem::assembly
