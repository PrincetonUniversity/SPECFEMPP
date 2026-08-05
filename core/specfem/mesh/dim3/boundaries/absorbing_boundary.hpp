#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh/mesh_base.hpp"
#include "specfem/mesh_entity.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief 3D absorbing boundary face information
 *
 * Stores the element indices and face identifiers for all boundary faces
 * classified as absorbing (Stacey) boundaries.  In the SPECFEM++ 3D mesh
 * format these are all faces whose direction is X_MIN, X_MAX, Y_MIN, Y_MAX,
 * or Z_MIN — i.e. every face that is not the top surface.
 */
template <> struct absorbing_boundary<specfem::element::dimension_tag::dim3> {

  constexpr static auto dimension =
      specfem::element::dimension_tag::dim3; ///< Dimension type

  int nelements = 0; ///< Number of absorbing boundary faces

  Kokkos::View<int *, Kokkos::HostSpace> index_mapping; ///< Spectral element
                                                        ///< index for each
                                                        ///< absorbing face

  Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
      type; ///< Which face of the element is on the absorbing boundary

  /**
   * @name Constructors
   */
  ///@{
  absorbing_boundary() = default;

  /**
   * @brief Allocate storage for @p nelements absorbing boundary faces.
   *
   * @param nelements Number of faces on the absorbing boundary
   */
  absorbing_boundary(const int nelements);
  ///@}
};

} // namespace mesh
} // namespace specfem
