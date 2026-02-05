#pragma once

#include "enumerations/interface.hpp"
#include "specfem/mesh/mesh_base.hpp"
#include "specfem/mesh_entity.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief Absorbing boundary information
 *
 * @tparam DimensionTag Dimension type for the mesh
 */
template <specfem::element::dimension_tag DimensionTag>
struct absorbing_boundary;

/**
 * @brief Absorbing boundary information
 *
 */
template <> struct absorbing_boundary<specfem::element::dimension_tag::dim2> {

  constexpr static auto dimension =
      specfem::element::dimension_tag::dim2; ///< Dimension
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
