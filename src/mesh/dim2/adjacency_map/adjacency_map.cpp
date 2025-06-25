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
bdtype_to_edge(const specfem::enums::boundaries::type &bd) {
  switch (bd) {
  case specfem::enums::boundaries::TOP:
    return specfem::enums::edge::type::TOP;
  case specfem::enums::boundaries::LEFT:
    return specfem::enums::edge::type::LEFT;
  case specfem::enums::boundaries::RIGHT:
    return specfem::enums::edge::type::RIGHT;
  case specfem::enums::boundaries::BOTTOM:
    return specfem::enums::edge::type::BOTTOM;
  default:
    return specfem::enums::edge::type::NONE;
  }
}
static inline specfem::enums::boundaries::type
edge_to_bdtype(const specfem::enums::edge::type &bd) {
  switch (bd) {
  case specfem::enums::edge::TOP:
    return specfem::enums::boundaries::type::TOP;
  case specfem::enums::edge::BOTTOM:
    return specfem::enums::boundaries::type::BOTTOM;
  case specfem::enums::edge::LEFT:
    return specfem::enums::boundaries::type::LEFT;
  case specfem::enums::edge::RIGHT:
    return specfem::enums::boundaries::type::RIGHT;
    break;
  default:
    // this should never be called
    return specfem::enums::boundaries::type::RIGHT;
  }
}
// ====================================================================
// helper: given an edge, returns either the clockwise or counterclockwise
// corner.
static inline specfem::enums::boundaries::type
edge_and_polarity_to_corner(const specfem::enums::edge::type &bd,
                            const bool counterclockwise) {
  switch (bd) {
  case specfem::enums::edge::TOP:
    return counterclockwise ? specfem::enums::boundaries::type::TOP_LEFT
                            : specfem::enums::boundaries::type::TOP_RIGHT;
  case specfem::enums::edge::BOTTOM:
    return counterclockwise ? specfem::enums::boundaries::type::BOTTOM_RIGHT
                            : specfem::enums::boundaries::type::BOTTOM_LEFT;
  case specfem::enums::edge::LEFT:
    return counterclockwise ? specfem::enums::boundaries::type::BOTTOM_LEFT
                            : specfem::enums::boundaries::type::TOP_LEFT;
  case specfem::enums::edge::RIGHT:
    return counterclockwise ? specfem::enums::boundaries::type::TOP_RIGHT
                            : specfem::enums::boundaries::type::BOTTOM_RIGHT;
    break;
  default:
    // this should never be called
    return specfem::enums::boundaries::type::RIGHT;
  }
}

/**
 * @brief calls set_as_boundary() for all boundary edges tagged by the
 * mesh::mesh.
 */
static inline void boundarymark(
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
    specfem::dimension::type::dim2>::adjacency_map()
    : nspec(-1) {}

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
    // this will never be called. Not throwing error for inlining.
    return 0;
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
  boundarymark(*this, mesh);

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
    std::set<std::pair<int, specfem::enums::boundaries::type> > &adj,
    std::list<std::pair<int, specfem::enums::boundaries::type> > &search,
    const int &ispec, const specfem::enums::boundaries::type &bdry) {
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
    std::set<std::pair<int, specfem::enums::boundaries::type> > &adj,
    std::list<std::pair<int, specfem::enums::boundaries::type> > &search,
    std::pair<int, specfem::enums::boundaries::type> &front) {
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
    std::set<std::pair<int, specfem::enums::boundaries::type> > &adj,
    std::list<std::pair<int, specfem::enums::boundaries::type> > &search,
    std::pair<int, specfem::enums::boundaries::type> &front) {
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

std::set<std::pair<int, specfem::enums::boundaries::type> >
specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>::
    get_all_conforming_adjacencies(
        const int ispec_start,
        const specfem::enums::boundaries::type bdry_start) const {
  std::set<std::pair<int, specfem::enums::boundaries::type> > adj;
  auto bd = std::make_pair(ispec_start, bdry_start);
  adj.insert(bd);

  // expand region by taking corner/edge adjacencies
  std::list<std::pair<int, specfem::enums::boundaries::type> > search;
  search.push_back(bd);

  while (!search.empty()) {
    bd = search.front();
    search.pop_front();
    switch (bd.second) {
    case enums::boundaries::type::TOP_LEFT:
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::TOP, false>(*this, adj, search, bd);
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::LEFT, true>(*this, adj, search, bd);
      break;
    case enums::boundaries::type::BOTTOM_LEFT:
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::LEFT, false>(*this, adj, search, bd);
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::BOTTOM, true>(*this, adj, search, bd);
      break;
    case enums::boundaries::type::BOTTOM_RIGHT:
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::BOTTOM, false>(*this, adj, search, bd);
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::RIGHT, true>(*this, adj, search, bd);
      break;
    case enums::boundaries::type::TOP_RIGHT:
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::RIGHT, false>(*this, adj, search, bd);
      get_all_conforming_adjacencies__test_corner<
          specfem::enums::edge::type::TOP, true>(*this, adj, search, bd);
      break;
    case enums::boundaries::type::TOP:
      get_all_conforming_adjacencies__test_edge<
          specfem::enums::edge::type::TOP>(*this, adj, search, bd);
      break;
    case enums::boundaries::type::LEFT:
      get_all_conforming_adjacencies__test_edge<
          specfem::enums::edge::type::LEFT>(*this, adj, search, bd);
      break;
    case enums::boundaries::type::RIGHT:
      get_all_conforming_adjacencies__test_edge<
          specfem::enums::edge::type::RIGHT>(*this, adj, search, bd);
      break;
    case enums::boundaries::type::BOTTOM:
      get_all_conforming_adjacencies__test_edge<
          specfem::enums::edge::type::BOTTOM>(*this, adj, search, bd);
      break;
    }
  }
  return adj;
}
