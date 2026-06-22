#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh/mesh_base.hpp"
#include "specfem/mesh_entity.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief 3D top-surface (free surface) face information
 *
 * Stores the element indices and face identifiers for all boundary faces
 * with direction Z_MAX (the top surface of the domain).  Acoustic elements
 * among these receive the @c acoustic_free_surface boundary tag; elastic
 * elements receive no special BC (free surface is the natural Neumann
 * condition for elastic).
 */
template <>
struct acoustic_free_surface<specfem::element::dimension_tag::dim3> {

  constexpr static auto dimension =
      specfem::element::dimension_tag::dim3; ///< Dimension type

  int nelem_acoustic_surface = 0; ///< Number of top-surface faces

  Kokkos::View<int *, Kokkos::HostSpace> index_mapping; ///< Spectral element
                                                        ///< index for each
                                                        ///< top-surface face

  Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>
      type; ///< Which face of the element is on the top surface

  /**
   * @name Constructors
   */
  ///@{
  acoustic_free_surface() = default;

  /**
   * @brief Allocate storage for @p n top-surface faces.
   *
   * @param n Number of top-surface faces
   */
  acoustic_free_surface(const int n);
  ///@}
};

} // namespace mesh
} // namespace specfem
