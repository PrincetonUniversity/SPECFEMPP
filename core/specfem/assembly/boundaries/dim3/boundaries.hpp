#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/macros.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {
/**
 * @brief Boundary condition information for every quadrature point in finite
 * element mesh
 *
 */
template <> struct boundaries<specfem::dimension::type::dim3> {

private:
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using BoundaryViewType =
      Kokkos::View<specfem::element::boundary_tag *,
                   Kokkos::DefaultExecutionSpace>; //< Underlying view
                                                   // type to store
                                                   // boundary tags

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag
  BoundaryViewType boundary_tags; ///< Boundary tags for every element in the
                                  ///< mesh
  BoundaryViewType::HostMirror h_boundary_tags; ///< Host mirror of boundary
                                                ///< tags

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
   * @param nglly Number of GLL points in y direction
   * @param ngllx Number of GLL points in x direction
   * @param mesh Finite element mesh information
   * @param mesh_assembly Finite element mesh assembly information
   * @param jacobian_matrix Jacobian matrix of basis functions at every
   * quadrature point
   */
  boundaries(
      const int nspec, const int ngllz, const int nglly, const int ngllx,
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
    const specfem::assembly::boundaries<specfem::dimension::type::dim3>
        &boundaries,
    PointBoundaryType &boundary) {

  constexpr auto tag = PointBoundaryType::boundary_tag;

  static_assert((tag == specfem::element::boundary_tag::none),
                "Boundary tag other than none is not yet supported.");

  if constexpr (tag == specfem::element::boundary_tag::none)
    return;

  IndexType l_index = index;

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
             const specfem::assembly::boundaries<specfem::dimension::type::dim3>
                 &boundaries,
             PointBoundaryType &boundary) {

  constexpr auto tag = PointBoundaryType::boundary_tag;

  static_assert((tag == specfem::element::boundary_tag::none),
                "Boundary tags other than none are not yet supported.");

  if constexpr (tag == specfem::element::boundary_tag::none)
    return;

  IndexType l_index = index;

  return;
}

} // namespace specfem::assembly
