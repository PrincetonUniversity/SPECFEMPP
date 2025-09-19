#pragma once

#include "absorbing_boundary.hpp"
#include "acoustic_free_surface.hpp"
#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief Boundary information
 *
 * @tparam DimensionTag Dimension type for the mesh
 */
template <> struct boundaries<specfem::dimension::type::dim3> {

  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension type

  specfem::mesh::absorbing_boundary<dimension_tag>
      absorbing_boundary; ///< Absorbing boundary
  specfem::mesh::acoustic_free_surface<dimension_tag>
      acoustic_free_surface; ///< Free surface

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
   * @brief Construct a new boundaries object
   *
   * @param absorbing_boundary absorbing boundary
   * @param acoustic_free_surface acoustic free surface
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
