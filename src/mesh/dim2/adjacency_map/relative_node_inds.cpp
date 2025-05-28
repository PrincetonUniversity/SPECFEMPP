
#include "mesh/dim2/adjacency_map/adjacency_map.hpp"

static inline void relative_node_inds(int &ix_adj, int &iz_adj, const int iedge,
                                      const int edgeind, const int edgeind_adj,
                                      const int ngll) {
#define EDGEIND_OF(x)                                                          \
  (specfem::mesh::adjacency_map::                                              \
       adjacency_map<specfem::dimension::type::dim2>::edge_to_index(           \
           specfem::enums::edge::type::x))
  switch (edgeind) {
  case EDGEIND_OF(RIGHT):
    switch (edgeind_adj) {
    case EDGEIND_OF(RIGHT):
      ix_adj = ngll - 1;
      iz_adj = ngll - 1 - iedge;
      return;
    case EDGEIND_OF(TOP):
      iz_adj = ngll - 1;
      ix_adj = iedge;
      return;
    case EDGEIND_OF(LEFT):
      ix_adj = 0;
      iz_adj = iedge;
      return;
    case EDGEIND_OF(BOTTOM):
      iz_adj = 0;
      ix_adj = ngll - 1 - iedge;
      return;
    }
  case EDGEIND_OF(TOP):
    switch (edgeind_adj) {
    case EDGEIND_OF(RIGHT):
      ix_adj = ngll - 1;
      iz_adj = iedge;
      return;
    case EDGEIND_OF(TOP):
      iz_adj = ngll - 1;
      ix_adj = ngll - 1 - iedge;
      return;
    case EDGEIND_OF(LEFT):
      ix_adj = 0;
      iz_adj = ngll - 1 - iedge;
      return;
    case EDGEIND_OF(BOTTOM):
      iz_adj = 0;
      ix_adj = iedge;
      return;
    }
  case EDGEIND_OF(LEFT):
    switch (edgeind_adj) {
    case EDGEIND_OF(RIGHT):
      ix_adj = ngll - 1;
      iz_adj = iedge;
      return;
    case EDGEIND_OF(TOP):
      iz_adj = ngll - 1;
      ix_adj = ngll - 1 - iedge;
      return;
    case EDGEIND_OF(LEFT):
      ix_adj = 0;
      iz_adj = ngll - 1 - iedge;
      return;
    case EDGEIND_OF(BOTTOM):
      iz_adj = 0;
      ix_adj = iedge;
      return;
    }
  case EDGEIND_OF(BOTTOM):
    switch (edgeind_adj) {
    case EDGEIND_OF(RIGHT):
      ix_adj = ngll - 1;
      iz_adj = ngll - 1 - iedge;
      return;
    case EDGEIND_OF(TOP):
      iz_adj = ngll - 1;
      ix_adj = iedge;
      return;
    case EDGEIND_OF(LEFT):
      ix_adj = 0;
      iz_adj = iedge;
      return;
    case EDGEIND_OF(BOTTOM):
      iz_adj = 0;
      ix_adj = ngll - 1 - iedge;
      return;
    }
  }
#undef EDGEIND_OF
}
