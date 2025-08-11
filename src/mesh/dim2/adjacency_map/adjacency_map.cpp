#include "enumerations/dimension.hpp"
#include "enumerations/specfem_enums.hpp"
#include "mesh/dim2/mesh.hpp"

#include "index_mapping_from_adjacencies.cpp"
#include <list>
#include <stdexcept>
#include <utility>

// ====================================================================
// conversions between edge and boundary enums.

static inline specfem::enums::edge::type
bdtype_to_edge(const specfem::mesh_entity::type &bd) {
  switch (bd) {
  case specfem::mesh_entity::type::top:
    return specfem::enums::edge::type::TOP;
  case specfem::mesh_entity::type::left:
    return specfem::enums::edge::type::LEFT;
  case specfem::mesh_entity::type::right:
    return specfem::enums::edge::type::RIGHT;
  case specfem::mesh_entity::type::bottom:
    return specfem::enums::edge::type::BOTTOM;
  default:
    return specfem::enums::edge::type::NONE;
  }
}
static inline specfem::mesh_entity::type
edge_to_bdtype(const specfem::enums::edge::type &bd) {
  switch (bd) {
  case specfem::enums::edge::TOP:
    return specfem::mesh_entity::type::top;
  case specfem::enums::edge::BOTTOM:
    return specfem::mesh_entity::type::bottom;
  case specfem::enums::edge::LEFT:
    return specfem::mesh_entity::type::left;
  case specfem::enums::edge::RIGHT:
    return specfem::mesh_entity::type::right;
    break;
  default:
    assert(false);
  }
}
// ====================================================================
// helper: given an edge, returns either the clockwise or counterclockwise
// corner.
static inline specfem::mesh_entity::type
edge_and_polarity_to_corner(const specfem::enums::edge::type &bd,
                            const bool counterclockwise) {
  switch (bd) {
  case specfem::enums::edge::TOP:
    return counterclockwise ? specfem::mesh_entity::type::top_left
                            : specfem::mesh_entity::type::top_right;
  case specfem::enums::edge::BOTTOM:
    return counterclockwise ? specfem::mesh_entity::type::bottom_right
                            : specfem::mesh_entity::type::bottom_left;
  case specfem::enums::edge::LEFT:
    return counterclockwise ? specfem::mesh_entity::type::bottom_left
                            : specfem::mesh_entity::type::top_left;
  case specfem::enums::edge::RIGHT:
    return counterclockwise ? specfem::mesh_entity::type::top_right
                            : specfem::mesh_entity::type::bottom_right;
    break;
  default:
    assert(false);
  }
}

/**
 * @brief calls set_as_boundary() for all boundary edges tagged by the
 * mesh::mesh.
 */
static inline void mark_all_boundaries(
    specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
        &adjmap,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &parent) {
  for (int i = 0; i < parent.boundaries.absorbing_boundary.nelements; i++) {
    const auto edgetype =
        bdtype_to_edge(parent.boundaries.absorbing_boundary.type(i));
    if (edgetype != specfem::enums::edge::type::NONE) {
      adjmap.set_as_boundary(
          parent.boundaries.absorbing_boundary.index_mapping(i), edgetype);
    }
  }
  for (int i = 0;
       i < parent.boundaries.acoustic_free_surface.nelem_acoustic_surface;
       i++) {
    const auto edgetype =
        bdtype_to_edge(parent.boundaries.acoustic_free_surface.type(i));
    if (edgetype != specfem::enums::edge::type::NONE) {
      adjmap.set_as_boundary(
          parent.boundaries.acoustic_free_surface.index_mapping(i), edgetype);
    }
  }
  // forcing boundary is never used, so skip it.
}

specfem::mesh::adjacency_map::adjacency_map<
    specfem::dimension::type::dim2>::adjacency_map(const int nspec)
    : nspec(nspec) {
  if (nspec >= 0) {
    adjacent_indices = decltype(adjacent_indices)(
        "specfem::compute::adjacency_map::adjacent_indices", nspec);
    adjacent_edges = decltype(adjacent_edges)(
        "specfem::compute::adjacency_map::adjacent_indices", nspec);
  }
}

static inline int node_edge(const specfem::enums::edge::type &edgetype,
                            const int &ind) {
  /* Control node layout:
   *  [3]━━[6]━━[2]
   *   ┃    ┆    ┃
   *  [7]╌╌[8]╌╌[5]
   *   ┃    ┆    ┃
   *  [0]━━[4]━━[1]
   */

  // indexed counter-clockwise
  switch (edgetype) {
  case specfem::enums::edge::type::TOP:
    return (ind == 0 ? 2 : (ind == 2 ? 3 : 6));
  case specfem::enums::edge::type::BOTTOM:
    return (ind == 0 ? 0 : (ind == 2 ? 1 : 4));
  case specfem::enums::edge::type::LEFT:
    return (ind == 0 ? 3 : (ind == 2 ? 0 : 7));
  case specfem::enums::edge::type::RIGHT:
    return (ind == 0 ? 1 : (ind == 2 ? 2 : 5));
  default:
    assert(false);
  }
}

/**
 * @brief Finds a conforming edge (shares node indices) among a list of candiate
 * elements
 * @return the (ispec, edgetype) pair of the corresponding edge. ispec == -1 if
 * none is found.
 */
template <int ngnod>
inline std::pair<int, specfem::enums::edge::type> find_conforming_edge(
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const int ispec, const specfem::enums::edge::type edge,
    const std::vector<int> &candidates);

template <>
inline std::pair<int, specfem::enums::edge::type> find_conforming_edge<4>(
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const int ispec, const specfem::enums::edge::type edge,
    const std::vector<int> &candidates) {
  // populate nodes_edge according to which edge this is
  const int nodes_edge_cw = mesh.control_nodes.knods(node_edge(edge, 0), ispec);
  const int nodes_edge_ccw =
      mesh.control_nodes.knods(node_edge(edge, 2), ispec);

  // iterate over candidates found by meshfem
  for (int ispec_o : candidates) {
    if (ispec_o > ispec) {
      // we we only need to iterate adjacencies once.
      continue;
    }

    for (auto edge_o : {
             specfem::enums::edge::type::RIGHT,
             specfem::enums::edge::type::TOP,
             specfem::enums::edge::type::LEFT,
             specfem::enums::edge::type::BOTTOM,
         }) {
      // adjacent element has same nodes, but in clockwise direction
      if (mesh.control_nodes.knods(node_edge(edge_o, 2), ispec_o) ==
              nodes_edge_cw &&
          mesh.control_nodes.knods(node_edge(edge_o, 0), ispec_o) ==
              nodes_edge_ccw) {

        return std::make_pair(ispec_o, edge_o);
      }
    }
  }
  return std::make_pair(-1, specfem::enums::edge::type::NONE);
}
template <>
inline std::pair<int, specfem::enums::edge::type> find_conforming_edge<9>(
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const int ispec, const specfem::enums::edge::type edge,
    const std::vector<int> &candidates) {

  int nodes_edge[3]; // store nodes along edge, in counterclockwise
                     // direction
  for (int i = 0; i < 3; i++) {
    nodes_edge[i] = mesh.control_nodes.knods(node_edge(edge, i), ispec);
  }

  // iterate over candidates found by meshfem
  for (int ispec_o : candidates) {
    if (ispec_o > ispec) {
      // we we only need to iterate adjacencies once.
      continue;
    }

    for (auto edge_o : {
             specfem::enums::edge::type::RIGHT,
             specfem::enums::edge::type::TOP,
             specfem::enums::edge::type::LEFT,
             specfem::enums::edge::type::BOTTOM,
         }) {
      // adjacent element has same nodes, but in clockwise direction
      bool matching = true;
      // adjacent element has same nodes, but in clockwise direction
      for (int i = 0; i < 3; i++) {
        if (i == 1 && mesh.control_nodes.ngnod == 4) {
          // no middle node for ngnod = 4
          continue;
        }
        if (nodes_edge[i] !=
            mesh.control_nodes.knods(node_edge(edge_o, 2 - i), ispec_o)) {
          matching = false;
          break;
        }
      }
      if (matching) {
        return std::make_pair(ispec_o, edge_o);
      }
    }
  }
  return std::make_pair(-1, specfem::enums::edge::type::NONE);
}

template <int ngnod>
static inline void mark_conforming(
    specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
        &adjmap,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const std::vector<std::vector<int> > &elements_with_shared_nodes) {
  static_assert(
      ngnod == 4 || ngnod == 9,
      "2d adjacency_map.cpp:mark_conforming(): ngnod must be either 4 or 9");
  const int &nspec = mesh.nspec;
  for (int ispec = 0; ispec < nspec; ispec++) {

    for (auto edge : {
             specfem::enums::edge::type::RIGHT,
             specfem::enums::edge::type::TOP,
             specfem::enums::edge::type::LEFT,
             specfem::enums::edge::type::BOTTOM,
         }) {
      const auto [ispec_o, edge_o] = find_conforming_edge<ngnod>(
          mesh, ispec, edge, elements_with_shared_nodes[ispec]);
      if (ispec_o != -1) {
        adjmap.create_conforming_adjacency(ispec, edge, ispec_o, edge_o);
      }
    }
  }
}
specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>::
    adjacency_map(
        const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
        const std::vector<std::vector<int> > &elements_with_shared_nodes)
    : nspec(mesh.nspec),
      adjacent_indices("specfem::compute::adjacency_map::adjacent_indices",
                       nspec),
      adjacent_edges("specfem::compute::adjacency_map::adjacent_indices",
                     nspec) {

  // null value: no adjacency
  for (int i = 0; i < nspec; i++) {
    for (int j = 0; j < 4; j++) {
      adjacent_edges(i, j) = specfem::enums::edge::type::NONE;
    }
  }
  // mark boundaries
  mark_all_boundaries(*this, mesh);

  //===================[ FIRST PASS ]===================
  // get conforming adjacencies from shared node indices

  switch (mesh.control_nodes.ngnod) {
  case 9:
    mark_conforming<9>(*this, mesh, elements_with_shared_nodes);
    break;
  case 4:
    mark_conforming<4>(*this, mesh, elements_with_shared_nodes);
    break;
  default:
    std::runtime_error("ngnod must be 4 or 9!");
  }
}

void specfem::mesh::adjacency_map::
    adjacency_map<specfem::dimension::type::dim2>::set_as_boundary(
        const int ispec, const specfem::enums::edge::type edge) {
  const int edgeind = edge_to_index(edge);
  adjacent_edges(ispec, edge) = specfem::enums::edge::type::NONE;
  adjacent_indices(ispec, edge) = -1;
}

bool specfem::mesh::adjacency_map::
    adjacency_map<specfem::dimension::type::dim2>::has_conforming_adjacency(
        const int ispec, const specfem::enums::edge::type edge) const {
  return has_conforming_adjacency(ispec, edge_to_index(edge));
}

bool specfem::mesh::adjacency_map::adjacency_map<
    specfem::dimension::type::dim2>::has_conforming_adjacency(const int ispec,
                                                              const int edge)
    const {
  return adjacent_edges(ispec, edge) != specfem::enums::edge::type::NONE &&
         adjacent_indices(ispec, edge) >= 0;
}

std::pair<int, specfem::enums::edge::type>
specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>::
    get_conforming_adjacency(const int ispec,
                             const specfem::enums::edge::type edge) const {
  const int edgeind = edge_to_index(edge);
  return std::make_pair(adjacent_indices(ispec, edgeind),
                        adjacent_edges(ispec, edgeind));
}
std::pair<int, specfem::enums::edge::type>
specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>::
    get_conforming_adjacency(const int ispec, const int edge) const {
  return std::make_pair(adjacent_indices(ispec, edge),
                        adjacent_edges(ispec, edge));
}

bool specfem::mesh::adjacency_map::
    adjacency_map<specfem::dimension::type::dim2>::has_boundary(
        const int ispec, const specfem::enums::edge::type edge) const {
  return has_boundary(ispec, edge_to_index(edge));
}

bool specfem::mesh::adjacency_map::adjacency_map<
    specfem::dimension::type::dim2>::has_boundary(const int ispec,
                                                  const int edge) const {
  return adjacent_edges(ispec, edge) == specfem::enums::edge::type::NONE &&
         adjacent_indices(ispec, edge) < 0;
}

void specfem::mesh::adjacency_map::
    adjacency_map<specfem::dimension::type::dim2>::create_conforming_adjacency(
        const int ispec1, const specfem::enums::edge::type edge1,
        const int ispec2, const specfem::enums::edge::type edge2) {
  int edgeind1 = edge_to_index(edge1);
  int edgeind2 = edge_to_index(edge2);
  adjacent_indices(ispec1, edgeind1) = ispec2;
  adjacent_edges(ispec1, edgeind1) = edge2;

  adjacent_indices(ispec2, edgeind2) = ispec1;
  adjacent_edges(ispec2, edgeind2) = edge1;
}

static inline void get_all_conforming_adjacencies__try_append(
    std::set<std::pair<int, specfem::mesh_entity::type> > &adj,
    std::list<std::pair<int, specfem::mesh_entity::type> > &search,
    const int &ispec, const specfem::mesh_entity::type &bdry) {
  /* helper for get_all_conforming_adjacencies:
   * main body for BFS -- if (ispec, bdry) was not yet explored, mark it and
   * append it to search.
   */
  const auto x = std::make_pair(ispec, bdry);
  if (adj.find(x) == adj.end()) {
    adj.insert(x);
    search.push_back(x);
  }
}

template <specfem::enums::edge::type edgetype, bool is_ccw_from_corner>
static inline void get_all_conforming_adjacencies__test_corner(
    const specfem::mesh::adjacency_map::adjacency_map<
        specfem::dimension::type::dim2> &map,
    std::set<std::pair<int, specfem::mesh_entity::type> > &adj,
    std::list<std::pair<int, specfem::mesh_entity::type> > &search,
    std::pair<int, specfem::mesh_entity::type> &front) {
  /* helper for get_all_conforming_adjacencies:
   * given a corner (specified by edge and direction), try_append the opposite
   * corner (across the given mating edge).
   */
  if (map.has_conforming_adjacency(
          front.first,
          specfem::mesh::adjacency_map::adjacency_map<
              specfem::dimension::type::dim2>::edge_to_index(edgetype))) {
    const auto other = map.get_conforming_adjacency(
        front.first,
        specfem::mesh::adjacency_map::adjacency_map<
            specfem::dimension::type::dim2>::edge_to_index(edgetype));
    get_all_conforming_adjacencies__try_append(
        adj, search, other.first,
        // edge is CCW from corner -> corner is not CCW side of edge
        // double negative from the polarity flip
        edge_and_polarity_to_corner(other.second, is_ccw_from_corner));
  }
}
template <specfem::enums::edge::type edgetype>
static inline void get_all_conforming_adjacencies__test_edge(
    const specfem::mesh::adjacency_map::adjacency_map<
        specfem::dimension::type::dim2> &map,
    std::set<std::pair<int, specfem::mesh_entity::type> > &adj,
    std::list<std::pair<int, specfem::mesh_entity::type> > &search,
    std::pair<int, specfem::mesh_entity::type> &front) {
  /* helper for get_all_conforming_adjacencies:
   * given a corner (specified by edge and direction), try_append the opposite
   * corner (across the given mating edge).
   */
  if (map.has_conforming_adjacency(
          front.first,
          specfem::mesh::adjacency_map::adjacency_map<
              specfem::dimension::type::dim2>::edge_to_index(edgetype))) {
    const auto other = map.get_conforming_adjacency(
        front.first,
        specfem::mesh::adjacency_map::adjacency_map<
            specfem::dimension::type::dim2>::edge_to_index(edgetype));
    get_all_conforming_adjacencies__try_append(adj, search, other.first,
                                               edge_to_bdtype(other.second));
  }
}

std::set<std::pair<int, specfem::mesh_entity::type> >
specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>::
    get_all_conforming_adjacencies(
        const int ispec_start,
        const specfem::mesh_entity::type bdry_start) const {
  // return value: fill `adj` with the conforming adjacencies
  std::set<std::pair<int, specfem::mesh_entity::type> > adj;

  // container for the boundary we are currently interested in. The first one is
  // the argument
  auto bd = std::make_pair(ispec_start, bdry_start);
  adj.insert(bd);

  /* expand "region" by taking corner/edge adjacencies. The conforming adjacency
   * graph for this geometry:
   * ┌───┐
   * │   ├───┬────┐
   * │ 0 │ 1 │  2 │
   * └───┼───🯚────┤
   *     🯗 3🯖 🯔 5 │
   *    🯖   🯗4🯕   │
   *
   * with element conforming adjacencies 1-2, 2-5, 5-4, 4-3, and 3-1 would have
   * conforming edge adjacencies:
   * - {1R, 2L}
   * - {2B, 5T}
   * - {5L, 4T}   (4 is assumed to be rotated clockwise 45°)
   * - {4L, 3R}
   * - {3T, 1B}
   *
   * The conforming corner adjacencies would include
   * - {1TR, 2TL}
   * - {1BL, 3TL} (0BR excluded, since 0R-1L is not conforming)
   * - {0BR}      (if 0B and 3L were conforming to a shared element, then this
   *               would be in the above set.)
   * - {1BR, 2BL, 5TL, 4TL, 3TR}
   */
  // this is BFS, but any algo would work, since we want a full traversal
  std::list<std::pair<int, specfem::mesh_entity::type> > search;
  search.push_back(bd);

  while (!search.empty()) {
    bd = search.front();
    search.pop_front();

    // we handle edges and corners differently. Corners first.
    switch (bd.second) {

    /*
     * for a given corner, search into the 2 corresponding edges. If they are
     * conforming, then the corresponding corner to the mating edge should also
     * be in adj.
     */
    case specfem::mesh_entity::type::top_left:
      // T is clockwise from TL
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::TOP, false>(*this, adj, search, bd);
      // L is counter-clockwise from TL
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::LEFT, true>(*this, adj, search, bd);
      break;
    case specfem::mesh_entity::type::bottom_left:
      // L is clockwise from BL
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::LEFT, false>(*this, adj, search, bd);
      // B is counter-clockwise from BL -- you get the idea
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::BOTTOM, true>(*this, adj, search, bd);
      break;
    case specfem::mesh_entity::type::bottom_right:
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::BOTTOM, false>(*this, adj, search, bd);
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::RIGHT, true>(*this, adj, search, bd);
      break;
    case specfem::mesh_entity::type::top_right:
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::RIGHT, false>(*this, adj, search, bd);
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::TOP, true>(*this, adj, search, bd);
      break;
    /*
     * for a given edge, we only need to check if the edge is conforming. If so,
     * add it to adj.
     */
    case specfem::mesh_entity::type::top:
      get_all_conforming_adjacencies__test_edge<
          specfem::enums::edge::type::TOP>(*this, adj, search, bd);
      break;
    case specfem::mesh_entity::type::left:
      get_all_conforming_adjacencies__test_edge<
          specfem::enums::edge::type::LEFT>(*this, adj, search, bd);
      break;
    case specfem::mesh_entity::type::right:
      get_all_conforming_adjacencies__test_edge<
          specfem::enums::edge::type::RIGHT>(*this, adj, search, bd);
      break;
    case specfem::mesh_entity::type::bottom:
      get_all_conforming_adjacencies__test_edge<
          specfem::enums::edge::type::BOTTOM>(*this, adj, search, bd);
      break;
    }
  }
  return adj;
}

std::string
specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>::
    edge_to_string(const int ispec, const specfem::enums::edge::type edge) {
  switch (edge) {
  case specfem::enums::edge::TOP:
    return std::to_string(ispec) + 'T';
  case specfem::enums::edge::BOTTOM:
    return std::to_string(ispec) + 'B';
  case specfem::enums::edge::LEFT:
    return std::to_string(ispec) + 'L';
  case specfem::enums::edge::RIGHT:
    return std::to_string(ispec) + 'R';
  default:
    return std::to_string(ispec);
  }
}

// std::string specfem::mesh::adjacency_map::adjacency_map<
//     specfem::dimension::type::dim2>::pretty_adjacency_table() {
//   std::ostringstream msg;
// #define COLWIDTH (7)
// #define NUMCOLS (5)
// #define NUM_DIGITS(st) (st < 10 ? 1 : (st < 100 ? 2 : (st < 1000 ? 3 : 4)))
//   char entry[COLWIDTH];
//   const auto print_entry = [&](bool terminate = false) {
//     int stsize;
//     for (stsize = 0; stsize < COLWIDTH && entry[stsize] != '\0'; stsize++) {
//     }
//     int padsize = COLWIDTH - stsize;
//     for (int i = padsize / 2; i > 0; i--) {
//       msg << ' ';
//     }
//     msg.write(entry, stsize);
//     for (int i = padsize - padsize / 2; i > 0; i--) {
//       msg << ' ';
//     }
//     if (terminate) {
//       msg << '\n';
//     } else {
//       msg << '|';
//     }
//   };

//   const auto set_entry_from_adj = [&](const int ispec,
//                                       const specfem::enums::edge::type type)
//                                       {
//     if (has_conforming_adjacency(ispec, type)) {
//       int ispec_adj;
//       specfem::enums::edge::type type_adj;
//       std::tie(ispec_adj, type_adj) = get_conforming_adjacency(ispec, type);
//       int padding_size = NUM_DIGITS(ispec_adj);
//       std::sprintf(
//           entry, "%d%c", ispec_adj,
//           type_adj == specfem::enums::edge::type::TOP
//               ? 'T'
//               : (type_adj == specfem::enums::edge::type::BOTTOM
//                      ? 'B'
//                      : (type_adj == specfem::enums::edge::type::LEFT ? 'L'
//                                                                      :
//                                                                      'R')));
//     } else if (has_boundary(ispec, type)) {
//       std::sprintf(entry, "(bdry)");
//     } else {
//       std::sprintf(entry, "MTR");
//     }
//   };
//   std::sprintf(entry, "ISPEC");
//   print_entry();
//   std::sprintf(entry, "RIGHT");
//   print_entry();
//   std::sprintf(entry, "TOP");
//   print_entry();
//   std::sprintf(entry, "LEFT");
//   print_entry();
//   std::sprintf(entry, "BOTTOM");
//   print_entry(true);
//   for (int i = 0; i < nspec; i++) {
//     std::sprintf(entry, "%d", i);
//     print_entry();
//     set_entry_from_adj(i, specfem::enums::edge::type::RIGHT);
//     print_entry();
//     set_entry_from_adj(i, specfem::enums::edge::type::TOP);
//     print_entry();
//     set_entry_from_adj(i, specfem::enums::edge::type::LEFT);
//     print_entry();
//     set_entry_from_adj(i, specfem::enums::edge::type::BOTTOM);
//     print_entry(true);
//   }
//   return msg.str();
// }
