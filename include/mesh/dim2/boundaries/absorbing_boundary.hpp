#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh_base.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief Absorbing boundary information
 *
 * @tparam DimensionTag Dimension type for the mesh
 */
template <specfem::dimension::type DimensionTag> struct absorbing_boundary;

/**
 * @brief Absorbing boundary information
 *
 */
template <> struct absorbing_boundary<specfem::dimension::type::dim2> {

  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension
                                      ///< type

  int nelements; ///< Number of elements on the absorbing boundary

  Kokkos::View<int *, Kokkos::HostSpace> index_mapping; ///< Spectral element
                                                        ///< index for elements
                                                        ///< on the absorbing
                                                        ///< boundary

  Kokkos::View<specfem::mesh_entity::dim2::type *, Kokkos::HostSpace>
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
  absorbing_boundary() {};

  absorbing_boundary(const int num_abs_boundaries_faces);

  ///@}
};
} // namespace mesh
} // namespace specfem
