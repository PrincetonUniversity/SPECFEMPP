#ifndef _MESH_CONTROL_NODES_HPP
#define _MESH_CONTROL_NODES_HPP

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"

namespace specfem {
namespace mesh {

struct control_nodes {
  int ngnod;                              ///< Number of control nodes
  int nspec;                              ///< Number of spectral elements
  specfem::kokkos::HostView2d<int> knods; ///< Global control element number for
                                          ///< every control node
  specfem::kokkos::HostView2d<type_real> coord; ///< (x, z) for every distinct
                                                ///< control node

  control_nodes() = default;

  control_nodes(const int &ndim, const int &nspec, const int &ngnod,
                const int &npgeo)
      : ngnod(ngnod), nspec(nspec),
        knods("specfem::mesh::control_nodes::knods", ngnod, nspec),
        coord("specfem::mesh::control_nodes::coord", ndim, npgeo) {}
};

} // namespace mesh
} // namespace specfem

#endif
