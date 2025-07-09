#pragma once

#include "Kokkos_Macros.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>
#include <set>
#include <string>

namespace specfem {
namespace mesh {
template <specfem::dimension::type Dimension> struct mesh;
namespace adjacency_map {

/**
 * @brief Stores the adjacencies between elements.
 *
 */
template <specfem::dimension::type Dimension> struct adjacency_map;
template <> struct adjacency_map<specfem::dimension::type::dim2> {

  /**
   * @brief an enumeration of the edges used by adjacency_map. Recovers the
   * index from an edge::type -- this is set explicitly since it must not
   * change, and some logic is simplified with this counter-clockwise ordering.
   */
  static constexpr int edge_to_index(const specfem::enums::edge::type edge) {
    assert(edge != specfem::enums::edge::NONE);
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
      return 0; // this is just to silence the linter; assert ensures this won't
                // be called.
    }
  }

  /**
   * @brief an enumeration of the edges used by adjacency_map. Recovers the
   * index from an edge::type -- this is set explicitly since it must not
   * change, and some logic is simplified with this counter-clockwise ordering.
   */
  static constexpr specfem::enums::edge::type edge_from_index(const int edge) {
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

  adjacency_map(const int nspec = -1);

  /**
   * @brief Construct a new adjacency map object from the mesh and the adjacency
   * structure built by meshfem.
   *
   * @param mesh - the mesh object parent
   * @param elements_with_shared_nodes - for each element, a list of elements
   * that are adjacent. This is generated from meshfem.
   */
  adjacency_map(
      const specfem::mesh::mesh<dimension::type::dim2> &mesh,
      const std::vector<std::vector<int> > &elements_with_shared_nodes);

  /**
   * @brief Checks if an edge of an element has a conforming adjacency (shared
   * node with respect to assembly).
   *
   * @param ispec - index of the element
   * @param edge - edge to check
   */
  bool has_conforming_adjacency(const int ispec,
                                const specfem::enums::edge::type edge) const;

  /**
   * @brief Checks if an edge of an element has a conforming adjacency (shared
   * node with respect to assembly).
   *
   * @param ispec - index of the element
   * @param edge - edge to check
   */
  bool has_conforming_adjacency(const int ispec, const int edge) const;

  /**
   * @brief For a conforming edge, returns its adjacenct edge. If the edge is
   * nonconforming, the behavior is undefined.
   *
   * @param ispec - index of the element
   * @param edge - edge to check
   * @return the pair (ispec, edge) of the mating edge.
   */
  std::pair<int, specfem::enums::edge::type>
  get_conforming_adjacency(const int ispec,
                           const specfem::enums::edge::type edge) const;
  /**
   * @brief For a conforming edge, returns its adjacenct edge. If the edge is
   * nonconforming, the behavior is undefined.
   *
   * @param ispec - index of the element
   * @param edge - edge to check
   * @return the pair (ispec, edge) of the mating edge.
   */
  std::pair<int, specfem::enums::edge::type>
  get_conforming_adjacency(const int ispec, const int edge) const;

  /**
   * @brief Returns whether or not the edge has a boundary. This will determine
   * if the edge will have a boundary tag. Due to how Neumann boundaries are
   * implemented, this will return false for them.
   *
   * @param ispec - index of the element
   * @param edge - edge to check
   */
  bool has_boundary(const int ispec,
                    const specfem::enums::edge::type edge) const;

  /**
   * @brief Returns whether or not the edge has a boundary. This will determine
   * if the edge will have a boundary tag. Due to how Neumann boundaries are
   * implemented, this will return false for them.
   *
   * @param ispec - index of the element
   * @param edge - edge to check
   */
  bool has_boundary(const int ispec, const int edge) const;

  /**
   * @brief Create a conforming adjacency between the to provided edges.
   *
   *
   * @param ispec1 - index of the first element
   * @param edge1 - edge of the first element
   * @param ispec2 - index of the second element
   * @param edge2 - edge of the second element
   */
  void create_conforming_adjacency(const int ispec1,
                                   const specfem::enums::edge::type edge1,
                                   const int ispec2,
                                   const specfem::enums::edge::type edge2);
  /**
   * @brief Marks this edge as a boundary edge. Boundary edges should not have
   * any adjacencies.
   *
   * @param ispec - index of the element
   * @param edge - edge to check
   */
  void set_as_boundary(const int ispec, const specfem::enums::edge::type edge);

  /**
   * @brief Builds a local to global index mapping that respects this adjacency
   * map.
   *
   * @param ngll - the number of quadrature points along an axis.
   * @return the pair (index_mapping, nglob) of the mapping and the number of
   * global degrees of freedom, respectively.
   */
  std::pair<specfem::kokkos::HostView3d<int>, int>
  generate_assembly_mapping(const int ngll) const;
  /**
   * @brief Recovers all of the conforming adjacencies to the given corner or
   * edge. When assembling, nodes corresponding to these adjacencies will share
   * global indices.
   *
   * @param ispec - index of the element
   * @param bdry - edge or corner of the element to check
   */
  std::set<std::pair<int, specfem::enums::boundaries::type> >
  get_all_conforming_adjacencies(
      const int ispec, const specfem::enums::boundaries::type bdry) const;

  /**
   * @brief Returns whether or not this adjacency map was built. If adjacency
   * data was not stored in the database, this will be false.
   */
  bool was_initialized() const { return nspec >= 0; }

  /**
   * @brief Returns a string representation of an edge as the element index
   * followed by a character code 'T', 'B', 'L', or 'R' representing the top,
   * bottom, left, or right (respectively) edge.
   *
   * @param ispec - index of the element
   * @param edge - edge type
   */
  static std::string edge_to_string(const int ispec,
                                    const specfem::enums::edge::type edge);

  /**
   * @brief Returns a string representation of this adjacency map as a table of
   * adjacencies.
   */
  // std::string pretty_adjacency_table();

private:
  int nspec;
  using IspecViewType =
      Kokkos::View<int *[4], Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using EdgeViewType = Kokkos::View<specfem::enums::edge::type *[4],
                                    Kokkos::LayoutLeft, Kokkos::HostSpace>;
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
};

} // namespace adjacency_map
} // namespace mesh
} // namespace specfem
