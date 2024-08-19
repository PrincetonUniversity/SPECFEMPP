#ifndef _ABSORBING_BOUNDARIES_HPP
#define _ABSORBING_BOUNDARIES_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
/**
 * Absorbing boundary conditions
 *
 * TODO : Document on how is this struct used in the code.
 */
struct absorbing_boundary {

  int nelements; ///< Number of elements on the absorbing boundary

  specfem::kokkos::HostView1d<int> ispec; ///< ispec value for the the element
                                          ///< on the boundary

  specfem::kokkos::HostView1d<specfem::enums::boundaries::type>
      type; ///< Type of the boundary

  /**
   * @brief Default constructor
   *
   */
  absorbing_boundary(){};

  /**
   * @brief Constructor to allocate views
   *
   * @param num_abs_boundaries_faces number of elements on absorbing boundary
   * face
   */
  absorbing_boundary(const int num_abs_boundaries_faces);

  /**
   * @brief Constructor to read fortran binary database.
   *
   * This constructor allocates views and assigns values to them read from the
   * database.
   *
   * @param stream Stream object for fortran binary file buffered to absorbing
   * boundaries section
   * @param num_abs_boundary_faces number of elements on absorbing boundary face
   * @param nspec Number of spectral elements
   * @param mpi Pointer to MPI object
   */
  absorbing_boundary(std::ifstream &stream, int num_abs_boundary_faces,
                     const int nspec, const specfem::MPI::MPI *mpi);
};
} // namespace mesh
} // namespace specfem

#endif
