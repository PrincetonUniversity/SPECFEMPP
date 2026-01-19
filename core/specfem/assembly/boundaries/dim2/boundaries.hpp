#pragma once

#include "enumerations/interface.hpp"
#include "impl/acoustic_free_surface.hpp"
#include "impl/stacey.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

/**
 * @brief Data container used to store information required to implement
 * boundary conditions in 2D SEM simulations
 *
 * This container provides constructors used to convert mesher-supplied boundary
 * to per-quadrature-point data required to implement different types of
 * boundary conditions (e.g., acoustic free surface, Stacey boundary conditions,
 * etc.), and data access functions to load/store data on device/host.
 */
template <> class boundaries<specfem::dimension::type::dim2> {

private:
  /**
   * @name Private Type Definitions
   *
   */
  ///@{
  /**
   * @brief Data type used to spectral element indices of elements on different
   * boundaries
   */
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  /**
   * @brief Data type used to store boundary tags for every element in the mesh
   *
   */
  using BoundaryViewType = Kokkos::View<specfem::element::boundary_tag *,
                                        Kokkos::DefaultExecutionSpace>;
  ///@}

public:
  /**
   * @name Public Constants
   *
   */
  ///@{
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension tag

  ///@}

  /**
   * @brief Device-accessible boundary tag information for every element in the
   * mesh
   *
   * This view stores the boundary tag for each spectral element in the mesh.
   * The boundary tags are used during device kernel execution to identify which
   * elements have specific boundary conditions.
   *
   * **Dimensions:** [nspec]
   *
   */
  BoundaryViewType boundary_tags;

  /**
   * @brief Host-accessible boundary tag information for every element in the
   * mesh
   *
   * This host mirror view stores the boundary tag for each spectral element in
   * the mesh. The boundary tags are used during host-side operations that
   * require knowledge of element boundary conditions.
   *
   * **Dimensions:** [nspec]
   */
  BoundaryViewType::HostMirror h_boundary_tags;

  /**
   * @brief Device-accessible mapping of spectral element index to acoustic free
   * surface
   *
   * This index mapping is used to access acoustic free surface boundary
   * specific information for a particular spectral element.
   *
   * **Dimensions:** [nspec]
   *
   * @code {.cpp}
   * // Example usage:
   * int spectral_element_index = 5; // Example spectral element index
   * int iz = 0, ix = 2; // GLL point indices
   * specfem::point::index<dimension_tag> index{spectral_element_index, iz, ix};
   * specfem::point::boundary<dimension_tag,
   * specfem::element::boundary_tag::acoustic_free_surface> boundary;
   * acoustic_free_surface.load_on_device(index, boundary);
   * @endcode
   *
   */
  IndexViewType acoustic_free_surface_index_mapping;

  /**
   * @brief Host-accessible mapping of spectral element index to acoustic free
   * surface
   *
   * This index mapping is used to access acoustic free surface boundary
   * specific information for a particular spectral element.
   *
   * **Dimensions:** [nspec]
   * @code {.cpp}
   * // Example usage:
   * int spectral_element_index = 5; // Example spectral element index
   * int iz = 0, ix = 2; // GLL point indices
   * specfem::point::index<dimension_tag> index{spectral_element_index, iz, ix};
   * specfem::point::boundary<dimension_tag,
   * specfem::element::boundary_tag::acoustic_free_surface> boundary;
   * acoustic_free_surface.load_on_device(index, boundary);
   * @endcode
   *
   */
  IndexViewType::HostMirror h_acoustic_free_surface_index_mapping;

  /**
   * @brief Device-accessible mapping of spectral element index to Stacey
   * boundary
   *
   * This index mapping is used to access Stacey boundary specific information
   * for a particular spectral element.
   *
   * **Dimensions:** [nspec]
   *
   * @code {.cpp}
   * // Example usage:
   * int spectral_element_index = 5; // Example spectral element index
   * int iz = 0, ix = 2; // GLL point indices
   * specfem::point::index<dimension_tag> index{spectral_element_index, iz, ix};
   * specfem::point::boundary<dimension_tag,
   * specfem::element::boundary_tag::stacey> boundary;
   * stacey.load_on_device(index, boundary);
   * @endcode
   *
   */
  IndexViewType stacey_index_mapping;

  /**
   * @brief Host-accessible mapping of spectral element index to Stacey
   * boundary
   *
   * This index mapping is used to access Stacey boundary specific information
   * for a particular spectral element.
   *
   * **Dimensions:** [nspec]
   *
   * @code {.cpp}
   * // Example usage:
   * int spectral_element_index = 5; // Example spectral element index
   * int iz = 0, ix = 2; // GLL point indices
   * specfem::point::index<dimension_tag> index{spectral_element_index, iz, ix};
   * specfem::point::boundary<dimension_tag,
   * specfem::element::boundary_tag::stacey> boundary;
   * stacey.load_on_device(index, boundary);
   * @endcode
   *
   */
  IndexViewType::HostMirror h_stacey_index_mapping;

  /**
   * @brief Data container used to store acoustic free surface boundary
   * information
   *
   */
  specfem::assembly::boundaries_impl::acoustic_free_surface<dimension_tag>
      acoustic_free_surface;

  /**
   * @brief Data container used to store Stacey boundary information
   *
   */
  specfem::assembly::boundaries_impl::stacey<dimension_tag> stacey;

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
   * @brief Contruct boundary condition data from mesh information.
   *
   * This constructor delegates to the individual boundary condition
   * constructors to construct per-quadrature-point data from mesher-supplied
   * boundary information.
   *
   * @param nspec Number of spectral elements in the mesh
   * @param ngllz Number of GLL points in the z (vertical) direction
   * @param ngllx Number of GLL points in the x (horizontal) direction
   * @param mesh Mesh containing boundary condition information
   * @param mesh_assembly Assembly mesh containing coordinate and connectivity
   * information
   * @param jacobian_matrix Jacobian matrix container with basis function
   * derivatives used for computing geometric transformations at quadrature
   * points
   */
  boundaries(
      const int nspec, const int ngllz, const int ngllx,
      const specfem::mesh::mesh<dimension_tag> &mesh,
      const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix);
  ///@}
};

/**
 * @defgroup BoundaryConditionDataAccess
 *
 */

/**
 * @brief Load boundary condition information for a quadrature point on device
 *
 * This function provides device-side access to boundary condition data by
 * loading appropriate boundary information based on the boundary tag associated
 * with a specific quadrature point.
 *
 * @ingroup BoundaryConditionDataAccess
 *
 * @tparam IndexType Index type for quadrature point location. Must be
 * @ref specfem::point::index with SIMD
 * usage matching PointBoundaryType
 * @tparam PointBoundaryType Point boundary data container. Must be @ref
 * specfem::point::boundary specialized for the appropriate boundary tag and
 * dimension
 *
 * @param index Quadrature point index specifying spectral element and GLL point
 * coordinates (ispec, iz, ix)
 * @param boundaries Container holding all boundary condition data for the mesh,
 * organized by boundary type
 * @param boundary Output parameter populated with boundary condition data for
 * the specified quadrature point
 *
 * @pre The boundary tag in PointBoundaryType must match the boundary tag stored
 * in `boundaries.boundary_tags[index.ispec]`
 * @pre IndexType and PointBoundaryType must have compatible SIMD configurations
 * (both SIMD or both non-SIMD)
 *
 * @note In debug builds, the function performs runtime validation of boundary
 * tag consistency and will abort execution if mismatched tags are detected
 *
 * @code
 * // Example usage in a device kernel
 * specfem::point::index<specfem::dimension::type::dim2> index{ispec, iz, ix};
 * specfem::point::boundary<specfem::element::boundary_tag::stacey,
 *                          specfem::dimension::type::dim2, false> boundary;
 *
 * // Load Stacey boundary data for mass matrix computation
 * specfem::assembly::load_on_device(index, assembly.boundaries, boundary);
 *
 * // Use loaded boundary data in physics calculations
 * specfem::boundary_conditions::apply_boundary_condition(..., boundary,
 * acceleration);
 * @endcode
 *
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
    if (boundaries.boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::acoustic_free_surface) {
      Kokkos::abort(
          "Boundary tag for acoustic free surface does not match the expected "
          "tag");
    }
#endif // NDEBUG
    l_index.ispec = boundaries.acoustic_free_surface_index_mapping(index.ispec);
    boundaries.acoustic_free_surface.load_on_device(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::stacey) {
#ifndef NDEBUG
    if (boundaries.boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::stacey) {
      Kokkos::abort(
          "Boundary tag for acoustic free surface does not match the expected "
          "tag");
    }
#endif // NDEBUG
    l_index.ispec = boundaries.stacey_index_mapping(index.ispec);
    boundaries.stacey.load_on_device(l_index, boundary);
  } else if constexpr (tag == specfem::element::boundary_tag::
                                  composite_stacey_dirichlet) {
#ifndef NDEBUG
    if (boundaries.boundary_tags(index.ispec) !=
        specfem::element::boundary_tag::composite_stacey_dirichlet) {
      Kokkos::abort(
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
 * @brief Load boundary condition information for a quadrature point on host
 *
 * This function provides host-side access to boundary condition data by
 * loading appropriate boundary information based on the boundary tag associated
 * with a specific quadrature point.
 *
 * @ingroup BoundaryConditionDataAccess
 *
 * @tparam IndexType Index type for quadrature point location. Must be
 * @ref specfem::point::index with SIMD
 * usage matching PointBoundaryType
 * @tparam PointBoundaryType Point boundary data container. Must be @ref
 * specfem::point::boundary specialized for the appropriate boundary tag and
 * dimension
 *
 * @param index Quadrature point index specifying spectral element and GLL point
 * coordinates (ispec, iz, ix)
 * @param boundaries Container holding all boundary condition data for the mesh,
 * organized by boundary type
 * @param boundary Output parameter populated with boundary condition data for
 * the specified quadrature point
 *
 * @pre The boundary tag in PointBoundaryType must match the boundary tag stored
 * in `boundaries.boundary_tags[index.ispec]`
 * @pre IndexType and PointBoundaryType must have compatible SIMD configurations
 * (both SIMD or both non-SIMD)
 *
 * @note In debug builds, the function performs runtime validation of boundary
 * tag consistency and will abort execution if mismatched tags are detected
 *
 * @code
 * // Example usage in a host function
 * specfem::point::index<specfem::dimension::type::dim2> index{ispec, iz, ix};
 * specfem::point::boundary<specfem::element::boundary_tag::stacey,
 *                          specfem::dimension::type::dim2, false> boundary;
 *
 * // Load Stacey boundary data for mass matrix computation
 * specfem::assembly::load_on_host(index, assembly.boundaries, boundary);
 *
 * // Use loaded boundary data in physics calculations
 * specfem::boundary_conditions::apply_boundary_condition(..., boundary,
 * acceleration);
 * @endcode
 *
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
      Kokkos::abort(
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
      Kokkos::abort(
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
      Kokkos::abort(
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
