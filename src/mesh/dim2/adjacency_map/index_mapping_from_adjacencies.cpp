#include "mesh/dim2/adjacency_map/adjacency_map.hpp"
#include "relative_node_inds.cpp"

std::pair<specfem::kokkos::HostView3d<int>, int>
specfem::mesh::adjacency_map::adjacency_map<
    specfem::dimension::type::dim2>::generate_assembly_mapping(const int ngll) {
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
#define inherit_adjacency_from_corner(bdtype)                                  \
  {                                                                            \
    bool found_value = false;                                                  \
    for (const auto &corners : get_all_conforming_adjacencies(                 \
             ispec, specfem::enums::boundaries::type::bdtype)) {               \
      ispec_adj = corners.first;                                               \
      if (corners.second == specfem::enums::boundaries::type::BOTTOM_LEFT) {   \
        ix_adj = 0;                                                            \
        iz_adj = 0;                                                            \
      } else if (corners.second ==                                             \
                 specfem::enums::boundaries::type::BOTTOM_RIGHT) {             \
        ix_adj = ngll - 1;                                                     \
        iz_adj = 0;                                                            \
      } else if (corners.second ==                                             \
                 specfem::enums::boundaries::type::TOP_RIGHT) {                \
        ix_adj = ngll - 1;                                                     \
        iz_adj = ngll - 1;                                                     \
      } else if (corners.second ==                                             \
                 specfem::enums::boundaries::type::TOP_LEFT) {                 \
        ix_adj = 0;                                                            \
        iz_adj = ngll - 1;                                                     \
      } else {                                                                 \
        continue;                                                              \
      }                                                                        \
      if ((iglob_adj = inds(ispec_adj, iz_adj, ix_adj)) != -1) {               \
        inds(ispec, iz, ix) = iglob_adj;                                       \
        found_value = true;                                                    \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
    if (found_value) {                                                         \
      continue;                                                                \
    }                                                                          \
  }
#define inherit_adjacency_from_edge(edgetype, iedge)                           \
  {                                                                            \
    if ((edge_adj = adjacent_edges(                                            \
             ispec, edge_to_index(specfem::enums::edge::type::edgetype))) !=   \
            specfem::enums::edge::type::NONE &&                                \
        (ispec_adj = adjacent_indices(                                         \
             ispec, edge_to_index(specfem::enums::edge::type::edgetype))) >=   \
            0) {                                                               \
      relative_node_inds(ix_adj, iz_adj, iedge,                                \
                         edge_to_index(specfem::enums::edge::type::edgetype),  \
                         edge_to_index(edge_adj), ngll);                       \
      if ((iglob_adj = inds(ispec_adj, iz_adj, ix_adj)) != -1) {               \
        inds(ispec, iz, ix) = iglob_adj;                                       \
        continue;                                                              \
      }                                                                        \
    }                                                                          \
  }
        if (ix == 0) {
          if (iz == 0) {
            inherit_adjacency_from_corner(BOTTOM_LEFT);
          } else if (iz == ngll - 1) {
            inherit_adjacency_from_corner(TOP_LEFT);
          } else {
            inherit_adjacency_from_edge(LEFT, iz);
          }
        } else if (ix == ngll - 1) {
          if (iz == 0) {
            inherit_adjacency_from_corner(BOTTOM_RIGHT);
          } else if (iz == ngll - 1) {
            inherit_adjacency_from_corner(TOP_RIGHT);
          } else {
            inherit_adjacency_from_edge(RIGHT, iz);
          }
        } else {
          if (iz == 0) {
            inherit_adjacency_from_edge(BOTTOM, ix);
          } else if (iz == ngll - 1) {
            inherit_adjacency_from_edge(TOP, ix);
          } else {
            // NO-OP -- we are on the interior, so no adjacency check
          }
        }
        inds(ispec, iz, ix) = nglob;
        nglob++;
      }
    }
  }
  return std::make_pair(inds, nglob);
}
