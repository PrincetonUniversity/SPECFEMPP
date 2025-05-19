#pragma once

#include "Kokkos_Macros.hpp"
#include "enumerations/dimension.hpp"
#include "mesh/dim2/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {
namespace adjacency_map {

struct nonconforming_edge {
  int edgeL, edgeR;
  type_real param_startL, param_startR;
  type_real param_endL, param_endR;

  nonconforming_edge() = default;
};

/**
 * @brief Stores the adjacencies between elements.
 *
 */
template <specfem::dimension::type DimensionTag> struct adjacency_map;

template <> struct adjacency_map<specfem::dimension::type::dim2> {
  static KOKKOS_INLINE_FUNCTION int
  edge_to_index(const specfem::enums::edge::type edge) {
    switch (edge) {
    case specfem::enums::edge::RIGHT:
      return 0;
    case specfem::enums::edge::TOP:
      return 1;
    case specfem::enums::edge::LEFT:
      return 2;
    case specfem::enums::edge::BOTTOM:
      return 3;
    default:
      return 0; // this should never be called
    }
  }
  static KOKKOS_INLINE_FUNCTION specfem::enums::edge::type
  edge_from_index(const int edge) {
    switch (edge) {
    case 0:
      return specfem::enums::edge::RIGHT;
    case 1:
      return specfem::enums::edge::TOP;
    case 2:
      return specfem::enums::edge::LEFT;
    case 3:
      return specfem::enums::edge::BOTTOM;
    default:
      return specfem::enums::edge::NONE;
    }
  }

  adjacency_map(
      const specfem::mesh::mesh<specfem::dimension::type::dim2> &parent);

  bool has_conforming_adjacency(const int ispec,
                                const specfem::enums::edge::type edge);

  bool has_conforming_adjacency(const int ispec, const int edge);

  std::pair<int,specfem::enums::edge::type> get_conforming_adjacency(const int ispec, const specfem::enums::edge::type edge);
  std::pair<int,specfem::enums::edge::type> get_conforming_adjacency(const int ispec, const int edge);

  bool has_boundary(const int ispec, const specfem::enums::edge::type edge);

  bool has_boundary(const int ispec, const int edge);

  void create_conforming_adjacency(const int ispec1,
                                   const specfem::enums::edge::type edge1,
                                   const int ispec2,
                                   const specfem::enums::edge::type edge2);

  void set_as_boundary(const int ispec, const specfem::enums::edge::type edge);

  void fill_nonconforming_adjacencies(
      const specfem::kokkos::HostView4d<double> &global_coordinates);

  static inline bool are_elements_conforming(
      const specfem::kokkos::HostView4d<double> &global_coordinates,
      const int ispec1, const specfem::enums::edge::type edge1,
      const int ispec2, const specfem::enums::edge::type edge2,
      type_real tolerance);

  const int &nspec;

private:
  const specfem::mesh::mesh<specfem::dimension::type::dim2> &parent;

  struct nonconforming_element_anchor {
    // element identifier
    int ispec;
    specfem::enums::edge::type edge;

    // nonconforming edge identifiers
    int edge_plus;  // edge at +1 local coordinate
    int edge_minus; // edge at -1 local coordinate

    bool side_plus;  // true if left side of edge at +1 local coordinate
    bool side_minus; // true if left side of edge at -1 local coordinate

    nonconforming_element_anchor()
        : ispec(-1), edge(specfem::enums::edge::type::NONE), edge_plus(-1),
          edge_minus(-1) {}
  };
  using IspecViewType =
      Kokkos::View<int *[4], Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using EdgeViewType = Kokkos::View<specfem::enums::edge::type *[4],
                                    Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using NonconformingElementAnchorViewType =
      Kokkos::View<nonconforming_element_anchor *, Kokkos::LayoutLeft,
                   Kokkos::HostSpace>;
  using NonconformingEdgeViewType =
      Kokkos::View<nonconforming_edge *, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  /* ---- adjacency storage ----
   * Each pair `(ispec, edge)` follows the following rule:
   * - If `adjacent_edges(ispec,edge)` is an edge (not NONE), then
   *   the `(adjecent_indices(ispec,edge), adjacent_edges(ispec,edge))` pair
   * shares a conforming edge to `(ispec, edge)`. Edges are conforming iff the
   * nodes along that edge share the same coordinates. This has no awareness of
   * any material decoupling.
   *
   * - If `adjacent_edges(ispec,edge)` is NONE, then the edge does not have a
   * conforming interface. In this case, if `adjecent_indices(ispec,edge)` is
   * negative, then this is a boundary, and no other edge shares an adjacency.
   * Otherwise, `adjecent_indices(ispec,edge)` is an index to <insert data
   * structure> for an interface.
   */
  IspecViewType adjacent_indices;
  EdgeViewType adjacent_edges;

  NonconformingElementAnchorViewType::HostMirror
      h_nonconforming_element_anchors;
};

} // namespace adjacency_map
} // namespace mesh
} // namespace specfem
