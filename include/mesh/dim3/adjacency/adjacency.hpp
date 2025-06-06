#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh_base.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store communication information between different MPI slices
 *
 * @tparam DimensionTag Dimension type
 */
template <> struct adjacency<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

  int nspec;             ///< Number of elements
  int num_neighbors_all; ///< Number of neighbors

  Kokkos::View<int *, Kokkos::HostSpace> neighbors_xadj; ///< adjacency offsets
  Kokkos::View<int *, Kokkos::HostSpace> neighbors_adjncy; ///< actual adjacency

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  adjacency() {};

  /**
   * @brief construct a container that tagged surface elements.
   *
   * @param nspec Number of elements
   * @param num_neighbors_all Number of neighbors
   *
   *
   */
  adjacency(const int nspec, const int num_neighbors_all)
      : nspec(nspec), num_neighbors_all(num_neighbors_all),
        neighbors_xadj("neighbors_xadj", nspec + 1),
        neighbors_adjncy("neighbors_adjncy", num_neighbors_all) {};

  ///@}
};

} // namespace mesh
} // namespace specfem
