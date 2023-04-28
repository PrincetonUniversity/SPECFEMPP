#ifndef _ABSORBING_BOUNDARIES_HPP
#define _ABSORBING_BOUNDARIES_HPP

#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
namespace boundaries {
/**
 * Absorbing boundary conditions
 *
 * TODO : Document on how is this struct used in the code.
 */
struct absorbing_boundary {

  specfem::kokkos::HostView1d<int> numabs; ///< ispec value for the the element
                                           ///< on the boundary

  specfem::kokkos::HostView1d<int> abs_boundary_type; ///< Defines if the
                                                      ///< absorbing boundary
                                                      ///< type is top, left,
                                                      ///< right or bottom. This
                                                      ///< is only used during
                                                      ///< plotting

  /**
   * @name Edge definitions
   *
   * ibegin_<edge#> defines the i or j index limits for loop iterations
   *
   */
  /// @{

  /**
   * @name Bottom boundary
   */
  /// @{
  specfem::kokkos::HostView1d<int> ibegin_edge1;
  specfem::kokkos::HostView1d<int> iend_edge1;
  /// @}

  /**
   * @name Right boundary
   */
  /// @{
  specfem::kokkos::HostView1d<int> ibegin_edge2;
  specfem::kokkos::HostView1d<int> iend_edge2;
  /// @}

  /**
   * @name Top boundary
   */
  /// @{
  specfem::kokkos::HostView1d<int> ibegin_edge3;
  specfem::kokkos::HostView1d<int> iend_edge3;
  /// @}

  /**
   * @name Left boundary
   */
  /// @{
  specfem::kokkos::HostView1d<int> ibegin_edge4;
  specfem::kokkos::HostView1d<int> iend_edge4;
  /// @}

  /// @}

  /**
   * @name Elements on boundary
   *
   * number of top/left/right/bottom elements on ith absorbing boundary
   *
   */
  ///@{
  specfem::kokkos::HostView1d<int> ib_bottom; ///< Number of bottom elements on
                                              ///< ith absorbing boundary
  specfem::kokkos::HostView1d<int> ib_top;    ///< Number of top elemetns on the
                                              ///< ith absobing boundary
  specfem::kokkos::HostView1d<int> ib_right;  ///< Number of right elemetns on
                                              ///< the ith absobing boundary
  specfem::kokkos::HostView1d<int> ib_left; ///< Number of left elemetns on the
                                            ///< ith absobing boundary
  ///@}
  /**
   * Specifies if an element is bottom, right, top or left absorbing boundary
   *
   * @code
   * for elements on bottom boundary
   *    codeabs(i, 0) == true
   * for elements on right boundary
   *    codeabs(i, 1) == true
   * for elements on top boundary
   *    codeabs(i, 2) == true
   * for elements on left boundary
   *    codeabs(i, 3) == true
   *@endcode
   *
   */
  specfem::kokkos::HostView2d<bool> codeabs;

  /**
   * Specifies if an element is bottom-left, bottom-right, top-left or top-right
   * corner element
   *
   * @code
   * for bottom-left boundary element
   *  codeabscorner(i, 0) == true
   * for bottom-right boundary element
   *  codeabscorner(i, 1) == true
   * for top-left boundary element
   *  codeabscorner(i, 2) == true
   * for top-right boundary element
   *  codeabscorner(i, 3) == true
   * @endcode
   */
  specfem::kokkos::HostView2d<bool> codeabscorner;

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
} // namespace boundaries
} // namespace mesh
} // namespace specfem

#endif
