#pragma once

#include "absorbing_boundary.hpp"
#include "acoustic_free_surface.hpp"
#include "specfem/element.hpp"
#include "specfem/mesh/mesh_base.hpp"

namespace specfem {
namespace mesh {

/**
 * @brief Boundary information for 3D spectral-element meshes
 *
 * Stores boundary face data from the SPECFEM++ 3D mesh database, split into
 * two distinct sub-structures that mirror the 2D boundary layout:
 *
 * - **absorbing_boundary**: faces whose direction is X_MIN, X_MAX, Y_MIN,
 *   Y_MAX, or Z_MIN.  Elements owning these faces receive the @c stacey
 *   boundary tag.
 * - **acoustic_free_surface**: faces whose direction is Z_MAX (top surface).
 *   Acoustic elements owning these faces receive the @c acoustic_free_surface
 *   boundary tag; elastic elements at Z_MAX get no special tag (free surface
 *   is the natural Neumann BC for elasticity).
 */
template <> struct boundaries<specfem::element::dimension_tag::dim3> {

  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension type

  specfem::mesh::absorbing_boundary<dimension_tag>
      absorbing_boundary; ///< Non-top absorbing boundary faces

  specfem::mesh::acoustic_free_surface<dimension_tag>
      acoustic_free_surface; ///< Top (Z_MAX) surface faces

  /**
   * @name Constructors
   */
  ///@{
  boundaries() = default;

  /**
   * @brief Construct from pre-populated sub-structures.
   *
   * @param absorbing_boundary Absorbing boundary faces (non-top directions)
   * @param acoustic_free_surface Top-surface faces (Z_MAX direction)
   */
  boundaries(const specfem::mesh::absorbing_boundary<dimension_tag>
                 &absorbing_boundary,
             const specfem::mesh::acoustic_free_surface<dimension_tag>
                 &acoustic_free_surface)
      : absorbing_boundary(absorbing_boundary),
        acoustic_free_surface(acoustic_free_surface) {}
  ///@}
};

} // namespace mesh
} // namespace specfem
