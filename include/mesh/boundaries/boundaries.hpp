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

  /**
   * @brief Constructor to read and assign values from fortran binary database
   * file
   *
   * @param stream Stream object for fortran binary file buffered to absorbing
   * boundary section
   * @param nspec Number of spectral elements
   * @param n_absorbing Number of absorbing boundary faces
   * @param n_acoustic_surface Number of acoustic free surface boundary faces
   * @param n_acforcing Number of acoustic forcing boundary faces
   * @param knods Spectral element control nodes
   * @param mpi Pointer to MPI object
   */
  boundaries(std::ifstream &stream, const int nspec, const int n_absorbing,
             const int n_acoustic_surface, const int n_acforcing,
             const Kokkos::View<int **, Kokkos::HostSpace> knods,
             const specfem::MPI::MPI *mpi)
      : absorbing_boundary(stream, n_absorbing, nspec, mpi),
        forcing_boundary(stream, n_acoustic_surface, nspec, mpi),
        acoustic_free_surface(stream, n_acforcing, knods, mpi){};
  ///@}
};
} // namespace mesh
} // namespace specfem
