#pragma once

#include "absorbing_boundaries.hpp"
#include "acoustic_free_surface.hpp"
#include "forcing_boundaries.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief
 *
 */
struct boundaries {
  specfem::mesh::absorbing_boundary absorbing_boundary; ///< Absorbing boundary
  specfem::mesh::acoustic_free_surface acoustic_free_surface; ///< Acoustic free
                                                              ///< surface
  specfem::mesh::forcing_boundary forcing_boundary; ///< Forcing boundary (never
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
  boundaries(const specfem::mesh::absorbing_boundary &absorbing_boundary,
             const specfem::mesh::acoustic_free_surface &acoustic_free_surface)
      : absorbing_boundary(absorbing_boundary),
        acoustic_free_surface(acoustic_free_surface) {}

  ///@}
};
} // namespace mesh
} // namespace specfem
