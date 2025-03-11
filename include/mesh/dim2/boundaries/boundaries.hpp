#pragma once

#include "absorbing_boundaries.hpp"
#include "acoustic_free_surface.hpp"
#include "enumerations/dimension.hpp"
#include "forcing_boundaries.hpp"
#include "mesh/mesh_base.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief Boundary information
 *
 * @tparam DimensionType Dimension type for the mesh
 */
template <specfem::dimension::type DimensionType> struct boundaries {

  constexpr static auto dimension = DimensionType; ///< Dimension type

  specfem::mesh::absorbing_boundary<DimensionType>
      absorbing_boundary; ///< Absorbing boundary
  specfem::mesh::acoustic_free_surface<DimensionType>
      acoustic_free_surface; ///< Acoustic free
                             ///< surface
  specfem::mesh::forcing_boundary<DimensionType> forcing_boundary; ///< Forcing
                                                                   ///< boundary
                                                                   ///< (never
                                                                   ///< used)

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
   * @param forcing_boundary forcing boundary
   */
  boundaries(
      const specfem::mesh::absorbing_boundary<DimensionType>
          &absorbing_boundary,
      const specfem::mesh::acoustic_free_surface<DimensionType>
          &acoustic_free_surface,
      const specfem::mesh::forcing_boundary<DimensionType> &forcing_boundary)
      : absorbing_boundary(absorbing_boundary),
        acoustic_free_surface(acoustic_free_surface),
        forcing_boundary(forcing_boundary) {}

  ///@}
};
} // namespace mesh
} // namespace specfem
