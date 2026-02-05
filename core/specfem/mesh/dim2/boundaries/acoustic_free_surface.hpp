#pragma once

#include "enumerations/interface.hpp"
#include "specfem/element.hpp"
#include "specfem/mesh_entity.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief Acoustic free surface boundary information
 *
 * @tparam DimensionTag Dimension type for the mesh
 */
template <specfem::element::dimension_tag DimensionTag>
struct acoustic_free_surface;

/**
 * @brief Acoustic free surface boundary information
 *
 */
template <>
struct acoustic_free_surface<specfem::element::dimension_tag::dim2> {

  constexpr static auto dimension =
      specfem::element::dimension_tag::dim2; ///< Dimension
                                             ///< type
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  acoustic_free_surface() {};

  acoustic_free_surface(const int nelem_acoustic_surface);

  ///@}

  int nelem_acoustic_surface; ///< Number of elements on the acoustic free
                              ///< surface boundary
  Kokkos::View<int *, Kokkos::HostSpace> index_mapping; ///< Spectral element
                                                        ///< index for elements
                                                        ///< on the acoustic
                                                        ///< free surface
                                                        ///< boundary
  Kokkos::View<specfem::mesh_entity::dim2::type *, Kokkos::HostSpace>
      type; ///< Which edge of the element is on the acoustic free surface
};
} // namespace mesh
} // namespace specfem
