#include "mesh/dim2/adjacency_map/adjacency_map.hpp"
#include "enumerations/specfem_enums.hpp"

#include "index_mapping_from_adjacencies.cpp"
#include <list>
#include <utility>

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

specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>::
    adjacency_map(
        const specfem::mesh::mesh<specfem::dimension::type::dim2> &parent)
    : parent(parent), nspec(parent.nspec),
      adjacent_indices("specfem::compute::adjacency_map::adjacent_indices",
                       nspec),
      adjacent_edges("specfem::compute::adjacency_map::adjacent_edges", nspec) {
  // null value: no adjacency
  for (int i = 0; i < nspec; i++) {
    for (int j = 0; j < 4; j++) {
      adjacent_edges(i, j) = specfem::enums::edge::type::NONE;
    }
  }

  // mark boundaries
  {
    for (int i = 0; i < parent.boundaries.absorbing_boundary.nelements; i++) {
      const auto edgetype =
          bdtype_to_edge(parent.boundaries.absorbing_boundary.type(i));
      if (edgetype != specfem::enums::edge::type::NONE) {
        set_as_boundary(parent.boundaries.absorbing_boundary.index_mapping(i),
                        edgetype);
      }
    }
    for (int i = 0;
         i < parent.boundaries.acoustic_free_surface.nelem_acoustic_surface;
         i++) {
      const auto edgetype =
          bdtype_to_edge(parent.boundaries.acoustic_free_surface.type(i));
      if (edgetype != specfem::enums::edge::type::NONE) {
        set_as_boundary(
            parent.boundaries.acoustic_free_surface.index_mapping(i), edgetype);
      }
    }
    // forcing boundary is never used, so skip it.
  }

  //===================[ FIRST PASS ]===================
  // get conforming adjacencies from shared node indices

  {
    std::vector<std::vector<int> > node_to_ispecs(
        parent.control_nodes.coord.extent(1));
    for (int ispec = 0; ispec < nspec; ispec++) {
      // check for edge adjacencies
      {
        int nodes_edge[3]; // store nodes along edge, in counterclockwise
                           // direction
#define node_bottom(ind) (ind == 0 ? 0 : (ind == 2 ? 1 : 4))
#define node_right(ind) (ind == 0 ? 1 : (ind == 2 ? 2 : 5))
#define node_top(ind) (ind == 0 ? 2 : (ind == 2 ? 3 : 6))
#define node_left(ind) (ind == 0 ? 3 : (ind == 2 ? 0 : 7))
#define node_edge(edgetype, ind)                                               \
  (edgetype == specfem::enums::edge::type::RIGHT                               \
       ? node_right(ind)                                                       \
       : (edgetype == specfem::enums::edge::type::TOP                          \
              ? node_top(ind)                                                  \
              : (edgetype == specfem::enums::edge::type::LEFT                  \
                     ? node_left(ind)                                          \
                     : node_bottom(ind))))
        for (auto edge : {
                 specfem::enums::edge::type::RIGHT,
                 specfem::enums::edge::type::TOP,
                 specfem::enums::edge::type::LEFT,
                 specfem::enums::edge::type::BOTTOM,
             }) {
          // did we find the candidate to this edge?
          bool matching = false;
          // populate nodes_edge according to which edge this is
          for (int i = 0; i < 3; i++) {
            nodes_edge[i] =
                parent.control_nodes.knods(node_edge(edge, i), ispec);
          }
          // candidates share the middle node
          for (int ispec_o : node_to_ispecs[nodes_edge[1]]) {

            for (auto edge_o : {
                     specfem::enums::edge::type::RIGHT,
                     specfem::enums::edge::type::TOP,
                     specfem::enums::edge::type::LEFT,
                     specfem::enums::edge::type::BOTTOM,
                 }) {
              matching = true;
              // adjacent element has same nodes, but in clockwise direction
              for (int i = 0; i < 3; i++) {
                if (nodes_edge[i] != parent.control_nodes.knods(
                                         node_edge(edge_o, 2 - i), ispec_o)) {
                  matching = false;
                  break;
                }
              }
              if (matching) {
                create_conforming_adjacency(ispec, edge, ispec_o, edge_o);
                break;
              }
            }
            if (matching) {
              break;
            }
          }
        }
      }
      // append this element
      for (int inod = 0; inod < parent.control_nodes.ngnod; inod++) {
        node_to_ispecs[parent.control_nodes.knods(inod, ispec)].push_back(
            ispec);
      }
    }
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
        const int ispec, const specfem::enums::edge::type edge) {
  return has_conforming_adjacency(ispec, edge_to_index(edge));
}

bool specfem::mesh::adjacency_map::adjacency_map<
    specfem::dimension::type::dim2>::has_conforming_adjacency(const int ispec,
                                                              const int edge) {
  return adjacent_edges(ispec, edge) != specfem::enums::edge::type::NONE &&
         adjacent_indices(ispec, edge) >= 0;
}

std::pair<int, specfem::enums::edge::type>
specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>::
    get_conforming_adjacency(const int ispec,
                             const specfem::enums::edge::type edge) {
  const int edgeind = edge_to_index(edge);
  return std::make_pair(adjacent_indices(ispec, edgeind),
                        adjacent_edges(ispec, edgeind));
}
std::pair<int, specfem::enums::edge::type>
specfem::mesh::adjacency_map::adjacency_map<
    specfem::dimension::type::dim2>::get_conforming_adjacency(const int ispec,
                                                              const int edge) {
  return std::make_pair(adjacent_indices(ispec, edge),
                        adjacent_edges(ispec, edge));
}

bool specfem::mesh::adjacency_map::
    adjacency_map<specfem::dimension::type::dim2>::has_boundary(
        const int ispec, const specfem::enums::edge::type edge) {
  return has_boundary(ispec, edge_to_index(edge));
}

bool specfem::mesh::adjacency_map::adjacency_map<
    specfem::dimension::type::dim2>::has_boundary(const int ispec,
                                                  const int edge) {
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

std::set<std::pair<int, specfem::enums::boundaries::type> >
specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>::
    get_all_conforming_adjacencies(
        const int ispec_start,
        const specfem::enums::boundaries::type bdry_start) {
  std::set<std::pair<int, specfem::enums::boundaries::type> > adj;
  auto bd = std::make_pair(ispec_start, bdry_start);
  adj.insert(bd);

  // expand region by taking corner/edge adjacencies
  std::list<std::pair<int, specfem::enums::boundaries::type> > search;
  search.push_back(bd);

  while (!search.empty()) {
    bd = search.front();
    search.pop_front();
#define TRY_APPEND(x)                                                          \
  {                                                                            \
    if (adj.find(x) == adj.end()) {                                            \
      adj.insert(x);                                                           \
      search.push_back(x);                                                     \
    }                                                                          \
  }
    switch (bd.second) {
    case enums::boundaries::type::TOP_LEFT:
#define TEST_CORNER(edge_, ccw_)                                               \
  {                                                                            \
    if (has_conforming_adjacency(                                              \
            bd.first, edge_to_index(specfem::enums::edge::type::edge_))) {     \
      const auto other = get_conforming_adjacency(                             \
          bd.first, edge_to_index(specfem::enums::edge::type::edge_));         \
      TRY_APPEND(std::make_pair(                                               \
          other.first, edge_and_polarity_to_corner(other.second, !ccw_)));     \
    }                                                                          \
  }
      TEST_CORNER(TOP, true);
      TEST_CORNER(LEFT, false);
      break;
    case enums::boundaries::type::BOTTOM_LEFT:
      TEST_CORNER(LEFT, true);
      TEST_CORNER(BOTTOM, false);
      break;
    case enums::boundaries::type::BOTTOM_RIGHT:
      TEST_CORNER(BOTTOM, true);
      TEST_CORNER(RIGHT, false);
      break;
    case enums::boundaries::type::TOP_RIGHT:
      TEST_CORNER(RIGHT, true);
      TEST_CORNER(TOP, false);
      break;
    case enums::boundaries::type::TOP:
#define TEST_EDGE(edge_)                                                       \
  {                                                                            \
    if (has_conforming_adjacency(                                              \
            bd.first, edge_to_index(specfem::enums::edge::type::edge_))) {     \
      const auto other = get_conforming_adjacency(                             \
          bd.first, edge_to_index(specfem::enums::edge::type::edge_));         \
      TRY_APPEND(std::make_pair(other.first, edge_to_bdtype(other.second)));   \
    }                                                                          \
  }
      TEST_EDGE(TOP);
      break;
    case enums::boundaries::type::LEFT:
      TEST_EDGE(LEFT);
      break;
    case enums::boundaries::type::RIGHT:
      TEST_EDGE(RIGHT);
      break;
    case enums::boundaries::type::BOTTOM:
      TEST_EDGE(BOTTOM);
      break;
    }
  }
  return adj;
}
