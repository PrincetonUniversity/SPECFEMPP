
#include "mesh/dim2/adjacency_map/adjacency_map.hpp"

static inline void relative_node_inds(int &ix_adj, int &iz_adj, const int iedge,
                                      const int edgeind, const int edgeind_adj,
                                      const int ngll) {
  switch (edgeind) {
  case specfem::mesh::adjacency_map::
      adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
          specfem::enums::edge::type::RIGHT):
    switch (edgeind_adj) {
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::RIGHT):
      ix_adj = ngll - 1;
      iz_adj = ngll - 1 - iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::TOP):
      iz_adj = ngll - 1;
      ix_adj = iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::LEFT):
      ix_adj = 0;
      iz_adj = iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::BOTTOM):
      iz_adj = 0;
      ix_adj = ngll - 1 - iedge;
      return;
    }
  case specfem::mesh::adjacency_map::
      adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
          specfem::enums::edge::type::TOP):
    switch (edgeind_adj) {
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::RIGHT):
      ix_adj = ngll - 1;
      iz_adj = iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::TOP):
      iz_adj = ngll - 1;
      ix_adj = ngll - 1 - iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::LEFT):
      ix_adj = 0;
      iz_adj = ngll - 1 - iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::BOTTOM):
      iz_adj = 0;
      ix_adj = iedge;
      return;
    }
  case specfem::mesh::adjacency_map::
      adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
          specfem::enums::edge::type::LEFT):
    switch (edgeind_adj) {
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::RIGHT):
      ix_adj = ngll - 1;
      iz_adj = iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::TOP):
      iz_adj = ngll - 1;
      ix_adj = ngll - 1 - iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::LEFT):
      ix_adj = 0;
      iz_adj = ngll - 1 - iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::BOTTOM):
      iz_adj = 0;
      ix_adj = iedge;
      return;
    }
  case specfem::mesh::adjacency_map::
      adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
          specfem::enums::edge::type::BOTTOM):
    switch (edgeind_adj) {
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::RIGHT):
      ix_adj = ngll - 1;
      iz_adj = ngll - 1 - iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::TOP):
      iz_adj = ngll - 1;
      ix_adj = iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::LEFT):
      ix_adj = 0;
      iz_adj = iedge;
      return;
    case specfem::mesh::adjacency_map::
        adjacency_map<specfem::dimension::type::dim2>::edge_to_index(
            specfem::enums::edge::type::BOTTOM):
      iz_adj = 0;
      ix_adj = ngll - 1 - iedge;
      return;
    }
  }
}
