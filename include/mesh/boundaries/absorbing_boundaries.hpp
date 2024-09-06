#pragma once

#include "enumerations/specfem_enums.hpp"
#include "specfem_mpi/specfem_mpi.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief Absorbing boundary information
 *
 */
struct absorbing_boundary {

  int nelements; ///< Number of elements on the absorbing boundary

  Kokkos::View<int *, Kokkos::HostSpace> index_mapping; ///< Spectral element
                                                        ///< index for elements
                                                        ///< on the absorbing
                                                        ///< boundary

  Kokkos::View<specfem::enums::boundaries::type *, Kokkos::HostSpace>
      type; ///< Which edge of the element is on the absorbing boundary

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  absorbing_boundary(){};

  absorbing_boundary(const int num_abs_boundaries_faces);

  /**
   * @brief Constructor to read and assign values from fortran binary database
   * file
   *
   * @param stream Stream object for fortran binary file buffered to absorbing
   * boundary section
   * @param num_abs_boundary_faces Number of absorbing boundary faces
   * @param nspec Number of spectral elements
   * @param mpi Pointer to MPI object
   */
  absorbing_boundary(std::ifstream &stream, int num_abs_boundary_faces,
                     const int nspec, const specfem::MPI::MPI *mpi);
  ///@}
};
} // namespace mesh
} // namespace specfem
