#ifndef _MESH_CONTROL_NODES_HPP
#define _MESH_CONTROL_NODES_HPP

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"

namespace specfem {
namespace mesh {

/**
 * @brief Control node information
 *
 */
struct control_nodes {
  using ViewType = Kokkos::View<type_real **, Kokkos::HostSpace>;

  int ngnod; ///< Number of control nodes
  int nspec; ///< Number of spectral elements
  Kokkos::View<int **, Kokkos::HostSpace> knods; ///< Control node indices
  ViewType coord; ///< Coordinates for control nodes

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default control_nodes constructor
   *
   */
  control_nodes() = default;

  /**
   * @brief Construct a new control_nodes object
   *
   * @param ndim Number of dimensions
   * @param nspec Number of spectral elements
   * @param ngnod Number of control nodes
   * @param npgeo Number of spectral element control nodes
   */
  control_nodes(const int &ndim, const int &nspec, const int &ngnod,
                const int &npgeo)
      : ngnod(ngnod), nspec(nspec),
        knods("specfem::mesh::control_nodes::knods", ngnod, nspec),
        coord("specfem::mesh::control_nodes::coord", ndim, npgeo) {}
  ///@}
};

} // namespace mesh
} // namespace specfem

#endif
