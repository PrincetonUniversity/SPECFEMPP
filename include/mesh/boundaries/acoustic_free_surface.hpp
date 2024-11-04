#pragma once

#include "enumerations/specfem_enums.hpp"
#include "specfem_mpi/specfem_mpi.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief Acoustic free surface boundary information
 *
 */
struct acoustic_free_surface {
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  acoustic_free_surface(){};

  acoustic_free_surface(const int nelem_acoustic_surface);
  /**
   * @brief Constructor to read and assign values from fortran binary database
   * file
   *
   * @param stream Stream object for fortran binary file buffered to absorbing
   * boundary section
   * @param nelem_acoustic_surface Number of absorbing boundary faces
   * @param knods Spectral element node connectivity
   * @param mpi Pointer to MPI object
   */
  acoustic_free_surface(std::ifstream &stream,
                        const int &nelem_acoustic_surface,
                        const Kokkos::View<int **, Kokkos::HostSpace> knods,
                        const specfem::MPI::MPI *mpi);
  ///@}

  int nelem_acoustic_surface; ///< Number of elements on the acoustic free
                              ///< surface boundary
  Kokkos::View<int *, Kokkos::HostSpace> index_mapping; ///< Spectral element
                                                        ///< index for elements
                                                        ///< on the acoustic
                                                        ///< free surface
                                                        ///< boundary
  Kokkos::View<specfem::enums::boundaries::type *, Kokkos::HostSpace>
      type; ///< Which edge of the element is on the acoustic free surface
};
} // namespace mesh
} // namespace specfem
