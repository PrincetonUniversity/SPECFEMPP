#include "mesh/dim2/adjacency_map/adjacency_map.hpp"
#include "relative_node_inds.cpp"

template <specfem::mesh_entity::type bdtype>
static inline void generate_assembly_mapping__push_adjacencies_to_corner(
    const specfem::mesh::adjacency_map::adjacency_map<
        specfem::dimension::type::dim2> &map,
    const specfem::kokkos::HostMirror3d<int> &inds, const int &ispec,
    const int &iz, const int &ix, const int &iglob_push, const int &ngll) {
  /* generate_assembly_mapping helper inline:
   *   handles iglob assignment for a given corner type
   *
   *   for adj in adjacency_map.all_adjacencies_to(ispec,iz,ix):
   *       set inds(adj) = iglob_push
   */
  int ix_adj, iz_adj, ispec_adj;

  for (const auto &corners :
       map.get_all_conforming_adjacencies(ispec, bdtype)) {
    // all of these will share our iglob
    ispec_adj = corners.first;
    if (corners.second == specfem::mesh_entity::type::bottom_left) {
      ix_adj = 0;
      iz_adj = 0;
    } else if (corners.second == specfem::mesh_entity::type::bottom_right) {
      ix_adj = ngll - 1;
      iz_adj = 0;
    } else if (corners.second == specfem::mesh_entity::type::top_right) {
      ix_adj = ngll - 1;
      iz_adj = ngll - 1;
    } else if (corners.second == specfem::mesh_entity::type::top_left) {
      ix_adj = 0;
      iz_adj = ngll - 1;
    } else {
      continue;
    }
    inds(ispec_adj, iz_adj, ix_adj) = iglob_push;
  }
}
template <specfem::enums::edge::type edgetype>
static inline void generate_assembly_mapping__push_adjacencies_to_edge(
    const specfem::mesh::adjacency_map::adjacency_map<
        specfem::dimension::type::dim2> &map,
    const specfem::kokkos::HostMirror3d<int> &inds, const int &ispec,
    const int &iz, const int &ix, const int &iglob_push, const int &ngll) {
  /* generate_assembly_mapping helper inline:
   *   handles iglob assignment for a given edge type
   *
   *   for adj in adjacency_map.all_adjacencies_to(ispec,iz,ix):
   *       set inds(adj) = iglob_push
   */
  int ix_adj, iz_adj, ispec_adj;
  specfem::enums::edge::type edge_adj;
  if (map.has_conforming_adjacency(ispec, edgetype)) {
    std::tie(ispec_adj, edge_adj) =
        map.get_conforming_adjacency(ispec, edgetype);
    relative_node_inds(ix_adj, iz_adj,
                       (edgetype == specfem::enums::edge::type::BOTTOM ||
                        edgetype == specfem::enums::edge::type::TOP)
                           ? ix
                           : iz,
                       map.edge_to_index(edgetype), map.edge_to_index(edge_adj),
                       ngll);
    inds(ispec_adj, iz_adj, ix_adj) = iglob_push;
  }
}

std::pair<specfem::kokkos::HostView3d<int>, int>
specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>::
    generate_assembly_mapping(const int ngll) const {
  kokkos::HostMirror3d<int> inds(
      "specfem::mesh::adjacency_map -- assembly_index_mapping", nspec, ngll,
      ngll);
  // initialize all to -1
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int ix = 0; ix < ngll; ix++) {
      for (int iz = 0; iz < ngll; iz++) {
        inds(ispec, iz, ix) = -1;
      }
    }
  }

  // global index numbering
  int nglob = 0;

  // temp vars for adjacent indices
  int ix_adj, iz_adj, ispec_adj, iglob_adj;
  specfem::enums::edge::type edge_adj;

  // assign
  for (int ix = 0; ix < ngll; ix++) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ispec = 0; ispec < nspec; ispec++) {
        if (inds(ispec, iz, ix) >= 0) {
          // already set this one through an adjacency in logic below.
          continue;
        }
        inds(ispec, iz, ix) = nglob;

        /*
         * Set inds(...) of local indices that are conformally adjacent to
         * (ispec,iz,ix). This is only for edge / corner nodes.
         */
        if (ix == 0) {
          if (iz == 0) {
            generate_assembly_mapping__push_adjacencies_to_corner<
                specfem::mesh_entity::type::bottom_left>(*this, inds, ispec, iz,
                                                         ix, nglob, ngll);
          } else if (iz == ngll - 1) {
            generate_assembly_mapping__push_adjacencies_to_corner<
                specfem::mesh_entity::type::top_left>(*this, inds, ispec, iz,
                                                      ix, nglob, ngll);
          } else {
            generate_assembly_mapping__push_adjacencies_to_edge<
                specfem::enums::edge::type::LEFT>(*this, inds, ispec, iz, ix,
                                                  nglob, ngll);
          }
        } else if (ix == ngll - 1) {
          if (iz == 0) {
            generate_assembly_mapping__push_adjacencies_to_corner<
                specfem::mesh_entity::type::bottom_right>(*this, inds, ispec,
                                                          iz, ix, nglob, ngll);
          } else if (iz == ngll - 1) {
            generate_assembly_mapping__push_adjacencies_to_corner<
                specfem::mesh_entity::type::top_right>(*this, inds, ispec, iz,
                                                       ix, nglob, ngll);
          } else {
            generate_assembly_mapping__push_adjacencies_to_edge<
                specfem::enums::edge::type::RIGHT>(*this, inds, ispec, iz, ix,
                                                   nglob, ngll);
          }
        } else {
          if (iz == 0) {
            generate_assembly_mapping__push_adjacencies_to_edge<
                specfem::enums::edge::type::BOTTOM>(*this, inds, ispec, iz, ix,
                                                    nglob, ngll);
          } else if (iz == ngll - 1) {
            generate_assembly_mapping__push_adjacencies_to_edge<
                specfem::enums::edge::type::TOP>(*this, inds, ispec, iz, ix,
                                                 nglob, ngll);
          }
        }
        nglob++;
      }
    }
  }
  return std::make_pair(inds, nglob);
}
