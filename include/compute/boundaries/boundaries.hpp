#ifndef _COMPUTE_BOUNDARIES_BOUNDARIES_HPP
#define _COMPUTE_BOUNDARIES_BOUNDARIES_HPP

#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/interface.hpp"
#include "impl/acoustic_free_surface.hpp"
#include "impl/stacey.hpp"
#include "macros.hpp"
#include "point/boundary.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace compute {
/**
 * @brief Boundary condition information for every quadrature point in finite
 * element mesh
 *
 */
struct boundaries {

private:
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using BoundaryViewType =
      Kokkos::View<specfem::element::boundary_tag *,
                   Kokkos::HostSpace>; //< Underlying view type to store
                                       // boundary tags

public:
  BoundaryViewType boundary_tags; ///< Boundary tags for every element in the
                                  ///< mesh

  IndexViewType acoustic_free_surface_index_mapping;
  IndexViewType::HostMirror h_acoustic_free_surface_index_mapping;

  IndexViewType stacey_index_mapping;
  IndexViewType::HostMirror h_stacey_index_mapping;

  specfem::compute::impl::boundaries::acoustic_free_surface
      acoustic_free_surface; ///< Acoustic free surface boundary

  specfem::compute::impl::boundaries::stacey stacey; ///< Stacey boundary

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
   * @param mapping mapping between mesh and compute indexing
   * @param quadrature Finite element quadrature information
   * @param properties Material properties for every quadrature point
   * @param partial_derivatives Partial derivatives of basis functions at every
   * quadrature point
   */
  boundaries(const int nspec, const int ngllz, const int ngllx,
             const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
             const specfem::compute::mesh_to_compute_mapping &mapping,
             const specfem::compute::quadrature &quadrature,
             const specfem::compute::properties &properties,
             const specfem::compute::partial_derivatives &partial_derivatives);
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
KOKKOS_FORCEINLINE_FUNCTION void
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
