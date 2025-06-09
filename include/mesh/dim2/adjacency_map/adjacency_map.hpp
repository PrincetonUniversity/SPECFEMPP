#pragma once

#include "Kokkos_Macros.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/specfem_enums.hpp"
#include <Kokkos_Core.hpp>
#include <set>

namespace specfem {
namespace mesh {
template <specfem::dimension::type dimtype> struct mesh;
namespace adjacency_map {

/**
 * @brief Stores the adjacencies between elements.
 *
 */
template <specfem::dimension::type dimtype> struct adjacency_map;
template <> struct adjacency_map<specfem::dimension::type::dim2> {
  static constexpr int edge_to_index(const specfem::enums::edge::type edge) {
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

  adjacency_map();
  adjacency_map(const specfem::mesh::mesh<dimension::type::dim2> &mesh);
  adjacency_map(
      const specfem::mesh::mesh<dimension::type::dim2> &mesh,
      const std::vector<std::vector<int> > &elements_with_shared_nodes);

  bool has_conforming_adjacency(const int ispec,
                                const specfem::enums::edge::type edge) const;

  bool has_conforming_adjacency(const int ispec, const int edge) const;

  std::pair<int, specfem::enums::edge::type>
  get_conforming_adjacency(const int ispec,
                           const specfem::enums::edge::type edge) const;
  std::pair<int, specfem::enums::edge::type>
  get_conforming_adjacency(const int ispec, const int edge) const;

  bool has_boundary(const int ispec,
                    const specfem::enums::edge::type edge) const;

  bool has_boundary(const int ispec, const int edge) const;

  void create_conforming_adjacency(const int ispec1,
                                   const specfem::enums::edge::type edge1,
                                   const int ispec2,
                                   const specfem::enums::edge::type edge2);

  void set_as_boundary(const int ispec, const specfem::enums::edge::type edge);

  std::pair<specfem::kokkos::HostView3d<int>, int>
  generate_assembly_mapping(const int ngll);
  std::set<std::pair<int, specfem::enums::boundaries::type> >
  get_all_conforming_adjacencies(
      const int ispec, const specfem::enums::boundaries::type bdry) const;

  bool was_initialized() { return nspec >= 0; }

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
