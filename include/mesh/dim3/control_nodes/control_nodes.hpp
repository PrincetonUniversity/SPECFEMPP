#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh_base.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::mesh {

template <> struct control_nodes<specfem::dimension::type::dim3> {

private:
  constexpr static int ndim = 3;

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag

  int nspec;  ///< Number of spectral elements
  int ngnod;  ///< Number of control nodes per spectral element
  int nnodes; ///< Total number of distinct control nodes

  using ControlNodeView = Kokkos::View<type_real **, Kokkos::LayoutLeft,
                                       Kokkos::DefaultHostExecutionSpace>;
  using IndexView = Kokkos::View<int **, Kokkos::LayoutLeft,
                                 Kokkos::DefaultHostExecutionSpace>;

  ControlNodeView coordinates; ///< Coordinates of control nodes for every
                               ///< spectral element
  IndexView index_mapping;     ///< Index mapping for control nodes

  control_nodes() = default;

  control_nodes(const int nspec, const int ngnod, const int nnodes)
      : nspec(nspec), ngnod(ngnod), nnodes(nnodes),
        index_mapping("control_nodes_indexing", nspec, ngnod),
        coordinates("control_nodes_coordinates", nnodes, ndim) {}

  std::string print() const;
};

} // namespace specfem::mesh
