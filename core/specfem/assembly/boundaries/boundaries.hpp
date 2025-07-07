#pragma once

#include "enumerations/interface.hpp"
#include "impl/acoustic_free_surface.hpp"
#include "impl/stacey.hpp"
#include "macros.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {
/**
 * @brief Boundary condition information for every quadrature point in finite
 * element mesh
 *
 */
template <> struct boundaries<specfem::dimension::type::dim2> {

private:
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using BoundaryViewType =
      Kokkos::View<specfem::element::boundary_tag *,
                   Kokkos::DefaultExecutionSpace>; //< Underlying view
                                                   // type to store
                                                   // boundary tags

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension tag
  BoundaryViewType boundary_tags; ///< Boundary tags for every element in the
                                  ///< mesh
  BoundaryViewType::HostMirror h_boundary_tags; ///< Host mirror of boundary
                                                ///< tags

  IndexViewType acoustic_free_surface_index_mapping; ///< Mapping of spectral
                                                     ///< element index to
                                                     ///< acoustic free surface
  IndexViewType::HostMirror h_acoustic_free_surface_index_mapping;

  IndexViewType stacey_index_mapping; ///< Mapping of spectral element index to
                                      ///< Stacey boundary
  IndexViewType::HostMirror h_stacey_index_mapping;

  specfem::assembly::boundaries_impl::acoustic_free_surface<dimension_tag>
      acoustic_free_surface; ///< Acoustic free surface boundary

  specfem::assembly::boundaries_impl::stacey<dimension_tag>
      stacey; ///< Stacey
              ///< boundary

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  boundaries() = default;

  /**
   * @brief Compute boundary conditions properties for every quadrature point in
   * the mesh
   *
   * @param nspec Number of spectral elements
   * @param ngllz Number of GLL points in z direction
   * @param ngllx Number of GLL points in x direction
   * @param mesh Finite element mesh information
   * @param mesh_assembly Finite element mesh assembly information
   * @param jacobian_matrix Jacobian matrix of basis functions at every
   * quadrature point
   */
  boundaries(const int nspec, const int ngllz, const int ngllx,
             const specfem::mesh::mesh<dimension_tag> &mesh,
             const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
             const specfem::assembly::jacobian_matrix &jacobian_matrix);
  ///@}
};

/**
 * @defgroup BoundaryConditionDataAccess
 *
 */

/**
 * @brief Load boundary condition information for a quadrature point on the
 * device
 *
 * @ingroup BoundaryConditionDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @tparam PointBoundaryType Point boundary type. Needs to be of @ref
 * specfem::point::boundary
 * @param index Index of the quadrature point
 * @param boundaries Boundary condition information for every quadrature point
 * @param boundary Boundary condition information for a given quadrature point
 * (output)
 */
template <typename IndexType, typename PointBoundaryType,
          typename std::enable_if<PointBoundaryType::simd::using_simd ==
                                      IndexType::using_simd,
                                  int>::type = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(
    const IndexType &index,
    const specfem::assembly::boundaries<specfem::dimension::type::dim2>
        &boundaries,
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
#ifndef NDEBUG
    if (boundaries.boundary_tag(index.ispec) !=
        specfem::element::boundary_tag::acoustic_free_surface) {
      Kokkos::Abort(
          "Boundary tag for acoustic free surface does not match the expected "
          "tag");
    }
#endif // NDEBUG
    l_index.ispec = boundaries.acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_device(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::stacey) {
#ifndef NDEBUG
    if (boundaries.boundary_tag(index.ispec) !=
        specfem::element::boundary_tag::stacey) {
      Kokkos::Abort(
          "Boundary tag for acoustic free surface does not match the expected "
          "tag");
    }
#endif // NDEBUG
    l_index.ispec = boundaries.stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_device(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::
                                  composite_stacey_dirichlet) {
#ifndef NDEBUG
    if (boundaries.boundary_tag(index.ispec) !=
        specfem::element::boundary_tag::composite_stacey_dirichlet) {
      Kokkos::Abort(
          "Boundary tag for acoustic free surface does not match the expected "
          "tag");
    }
#endif // NDEBUG
    l_index.ispec = boundaries.acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_device(l_index, boundary);
    l_index.ispec = boundaries.stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_device(l_index, boundary);
  }

  return;
}

/**
 * @brief Load boundary condition information for a quadrature point on the host
 *
 * @ingroup BoundaryConditionDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @tparam PointBoundaryType Point boundary type. Needs to be of @ref
 * specfem::point::boundary
 * @param index Index of the quadrature point
 * @param boundaries Boundary condition information for every quadrature point
 * @param boundary Boundary condition information for a given quadrature point
 * (output)
 */
template <typename IndexType, typename PointBoundaryType,
          typename std::enable_if<PointBoundaryType::simd::using_simd ==
                                      IndexType::using_simd,
                                  int>::type = 0>
inline void
load_on_host(const IndexType &index,
             const specfem::assembly::boundaries<specfem::dimension::type::dim2>
                 &boundaries,
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
#ifndef NDEBUG
    if (boundaries.h_boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::acoustic_free_surface) {
      throw std::runtime_error(
          "Boundary tag for acoustic free surface does not match the expected "
          "tag");
    }
#endif // NDEBUG
    l_index.ispec =
        boundaries.h_acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_host(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::stacey) {
#ifndef NDEBUG
    if (boundaries.h_boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::stacey) {
      throw std::runtime_error(
          "Boundary tag for acoustic free surface does not match the expected "
          "tag");
    }
#endif // NDEBUG
    l_index.ispec = boundaries.h_stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_host(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::
                                  composite_stacey_dirichlet) {
#ifndef NDEBUG
    if (boundaries.h_boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::composite_stacey_dirichlet) {
      throw std::runtime_error(
          "Boundary tag for acoustic free surface does not match the expected "
          "tag");
    }
#endif // NDEBUG
    l_index.ispec =
        boundaries.h_acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_host(l_index, boundary);
    l_index.ispec = boundaries.h_stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_host(l_index, boundary);
  }

  return;
}

} // namespace specfem::assembly
