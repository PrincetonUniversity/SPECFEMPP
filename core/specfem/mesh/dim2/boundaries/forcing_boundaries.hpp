#ifndef _FORCING_BOUNDARIES_HPP
#define _FORCING_BOUNDARIES_HPP

#include "enumerations/dimension.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace mesh {
/**
 * @brief Forcing boundary information
 *
 * @tparam DimensionTag Dimension type for the mesh
 */
template <specfem::dimension::type DimensionTag> struct forcing_boundary;

/**
 * @brief Forcing boundary information
 *
 */
template <> struct forcing_boundary<specfem::dimension::type::dim2> {

  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension
                                      ///< type

  specfem::kokkos::HostView1d<int> numacforcing;  ///< ispec value for the the
                                                  ///< element on the boundary
  specfem::kokkos::HostView1d<int> typeacforcing; ///< Defines if the acoustic
                                                  ///< forcing boundary type is
                                                  ///< top, left, right or
                                                  ///< bottom. This is only used
                                                  ///< during plotting
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
   * number of top/left/right/bottom elements on ith acoustic forcing boundary
   *
   */
  ///@{
  specfem::kokkos::HostView1d<int> ib_bottom; ///< Number of bottom elements on
                                              ///< ith acoustic forcing boundary
  specfem::kokkos::HostView1d<int> ib_top;    ///< Number of top elemetns on the
                                              ///< ith acoustic forcing boundary
  specfem::kokkos::HostView1d<int> ib_right;  ///< Number of right elemetns on
                                              ///< the ith acoustic forcing
                                              ///< boundary
  specfem::kokkos::HostView1d<int> ib_left; ///< Number of left elemetns on the
                                            ///< ith acoustic forcing boundary
  ///@}
  /**
   * Specifies if an element is bottom, right, top or left absorbing boundary
   *
   * @code
   * for elements on bottom boundary
   *    codeacforcing(i, 0) == true
   * for elements on right boundary
   *    codeacforcing(i, 1) == true
   * for elements on top boundary
   *    codeacforcing(i, 2) == true
   * for elements on left boundary
   *    codeacforcing(i, 3) == true
   *@endcode
   *
   */
  specfem::kokkos::HostView2d<bool> codeacforcing;

  /**
   * @brief Default constructor
   *
   */
  forcing_boundary() {};
  /**
   * @brief Constructor to allocate views
   *
   * @param nelement_acforcing number of elements on acoustic forcing boundary
   * face
   */
  forcing_boundary(const int nelement_acforcing);
  /**
   * @brief Constructor to read fortran binary database.
   *
   * This constructor allocates views and assigns values to them read from the
   * database.
   *
   * @param stream Stream object for fortran binary file buffered to absorbing
   * boundaries section
   * @param nelement_acforcing number of elements on absorbing boundary face
   * @param nspec Number of spectral elements
   */
  forcing_boundary(std::ifstream &stream, const int nelement_acforcing,
                   const int nspec);
};

} // namespace mesh
} // namespace specfem

#endif
