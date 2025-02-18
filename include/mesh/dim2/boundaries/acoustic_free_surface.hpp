#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/specfem_enums.hpp"
#include "specfem_mpi/specfem_mpi.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief Acoustic free surface boundary information
 *
 * @tparam DimensionType Dimension type for the mesh
 */
template <specfem::dimension::type DimensionType> struct acoustic_free_surface;

/**
 * @brief Acoustic free surface boundary information
 *
 */
template <> struct acoustic_free_surface<specfem::dimension::type::dim2> {

  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension
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
  acoustic_free_surface(){};

  acoustic_free_surface(const int nelem_acoustic_surface);

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
