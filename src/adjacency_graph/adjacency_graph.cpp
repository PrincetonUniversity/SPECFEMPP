#ifndef _ADJACENCY_GRAPH_CPP_
#define _ADJACENCY_GRAPH_CPP_

#include "adjacency_graph/adjacency_graph.hpp"

namespace specfem {
namespace adjacency_graph {

specfem::adjacency_graph::adjacency_graph<4> from_index_mapping(
    const Kokkos::View<int ***, Kokkos::LayoutLeft,
                       Kokkos::DefaultExecutionSpace> &index_mapping) {

  int nspec = index_mapping.extent(0);
  int ngllz = index_mapping.extent(1);
  int ngllx = index_mapping.extent(2);

  if (ngllz < 3 || ngllx < 3) {
    throw std::runtime_error("specfem::adjacency_graph::from_index_mapping() "
                             "relies on >= 3 quadrature points per axis");
  }

  int nglob = compute_nglob(index_mapping);

  specfem::adjacency_graph::adjacency_graph<4> graph(nspec);

  std::vector<specfem::adjacency_graph::adjacency_pointer> inv_index(
      nglob, specfem::adjacency_graph::adjacency_pointer());

  const auto set_ref_inds = [&](int &ix, int &iz, const int8_t side,
                                const bool flip, const int off = 1) {
    int fac = (side / 2 == 0) ? 1 : 0;
    if ((side % 2) == 0) { // +/- x
      iz = flip ? (ngllz - 2) : 1;
      ix = fac * (ngllx - 1);
    } else {
      ix = flip ? (ngllx - 2) : 1;
      iz = fac * (ngllz - 1);
    }
  };
  int aaa = 0;
  const auto verify_adjacencies = [&](const int ispec, const int side) {
    int ix, iz;
    set_ref_inds(ix, iz, side, false);
    int iglob = index_mapping(ispec, iz, ix);
    if (!inv_index[iglob].is_active()) {
      // not yet defined, so define it (symmetrically to capture index flips)
      inv_index[iglob] =
          specfem::adjacency_graph::adjacency_pointer(ispec, side, false);
      set_ref_inds(ix, iz, side, true);
      inv_index[iglob] =
          specfem::adjacency_graph::adjacency_pointer(ispec, side, false);
    } else {
      int ispec_ = inv_index[iglob].elem;
      int side_ = inv_index[iglob].side;
      // if((ispec == 0 && ispec_ == 74) || (ispec == 74 && ispec_ == 0)){
      // //if(ispec == 0 || ispec_ == 0){
      //   graph.form_adjacency(ispec,side,ispec_,side_,false);
      // }
      // this matches another point, so set adjacency. did we flip?
      set_ref_inds(ix, iz, side, false, 0);
      int iglob_corner_L = index_mapping(ispec, iz, ix);
      set_ref_inds(ix, iz, side, true, 0);
      int iglob_corner_H = index_mapping(ispec, iz, ix);
      set_ref_inds(ix, iz, side_, false, 0);
      int iglob_corner_L_ = index_mapping(ispec_, iz, ix);
      set_ref_inds(ix, iz, side_, true, 0);
      int iglob_corner_H_ = index_mapping(ispec_, iz, ix);
      if (iglob_corner_L == iglob_corner_L_ &&
          iglob_corner_H == iglob_corner_H_) {
        // no flip
        graph.form_adjacency(ispec, side, ispec_, side_, false);
      } else if (iglob_corner_L == iglob_corner_H_ &&
                 iglob_corner_H == iglob_corner_L_) {
        // flip
        graph.form_adjacency(ispec, side, ispec_, side_, true);
      } else {
        throw std::runtime_error(
            "specfem::adjacency_graph::from_index_mapping() - error "
            "encountered during index assignment.");
      }
    }
  };
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int edge = 0; edge < 4; edge++) {
      verify_adjacencies(ispec, edge);
    }
  }
  return graph;
}

static bool _reflect(const specfem::adjacency_graph::adjacency_graph<4> &graph,
                     int &ispec_set, int &ix_set, int &iz_set, const int &ispec,
                     const int &ix, const int &iz, const int8_t &side,
                     const int &ngllz, const int &ngllx) {
  specfem::adjacency_graph::adjacency_pointer adj =
      graph.get_adjacency(ispec, side);
  ispec_set = adj.elem;

  bool self_sign = (side / 2) == 0;
  bool other_x_align = (adj.side == 0 || adj.side == 2);

  if ((side - adj.side + 2) % 2 == 0) {
    ix_set = ix;
    iz_set = iz;
  } else { // swap x and y
    ix_set = iz;
    iz_set = ix;
  }

  // flip x only if sign changes (if other is +/- x); else look at flip bit
  if (other_x_align ? ((adj.side == 2) == self_sign) : adj.flip)
    ix_set = ngllx - ix_set - 1;

  // flip z only if sign changes (if other is +/- z); else look at flip bit
  if (other_x_align ? adj.flip : ((adj.side == 3) == self_sign))
    iz_set = ngllz - iz_set - 1;

  return adj.is_active();
}

static void _propagate_index(
    const specfem::adjacency_graph::adjacency_graph<4> &graph,
    Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
        &index_mapping,
    const int &ispec, const int &ix, const int &iz, const int &index,
    const int &ngllz, const int &ngllx) {
  // set this index, then propagate through all edges that this node is on.
  int ispec_, ix_, iz_;
  index_mapping(ispec, iz, ix) = index;
  if (ix == ngllx - 1 &&
      _reflect(graph, ispec_, ix_, iz_, ispec, ix, iz, 0, ngllz, ngllx) &&
      index_mapping(ispec_, iz_, ix_) == -1) {
    _propagate_index(graph, index_mapping, ispec_, ix_, iz_, index, ngllz,
                     ngllx);
  } else if (ix == 0 &&
             _reflect(graph, ispec_, ix_, iz_, ispec, ix, iz, 2, ngllz,
                      ngllx) &&
             index_mapping(ispec_, iz_, ix_) == -1) {
    _propagate_index(graph, index_mapping, ispec_, ix_, iz_, index, ngllz,
                     ngllx);
  } else if (iz == ngllz - 1 &&
             _reflect(graph, ispec_, ix_, iz_, ispec, ix, iz, 1, ngllz,
                      ngllx) &&
             index_mapping(ispec_, iz_, ix_) == -1) {
    _propagate_index(graph, index_mapping, ispec_, ix_, iz_, index, ngllz,
                     ngllx);
  }
  if (iz == 0 &&
      _reflect(graph, ispec_, ix_, iz_, ispec, ix, iz, 3, ngllz, ngllx) &&
      index_mapping(ispec_, iz_, ix_) == -1) {
    _propagate_index(graph, index_mapping, ispec_, ix_, iz_, index, ngllz,
                     ngllx);
  }
}

Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
to_index_mapping(const specfem::adjacency_graph::adjacency_graph<4> &graph,
                 const int ngllz, const int ngllx, int *nglob_set) {

  int nspec = graph.get_size();
  Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
      index_mapping("specfem::adjacency_graph::to_index_mapping()", nspec,
                    ngllz, ngllx);
  for (int ix = 0; ix < ngllx; ix++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ispec = 0; ispec < nspec; ispec++) {
        index_mapping(ispec, iz, ix) = -1;
      }
    }
  }
  int count = 0;

  int ispec_, ix_, iz_, ispec_refl, ix_refl, iz_refl;
  for (int ix = 0; ix < ngllx; ix++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ispec = 0; ispec < nspec; ispec++) {
        if (index_mapping(ispec, iz, ix) == -1) {
          // this class of points was not set yet
          _propagate_index(graph, index_mapping, ispec, ix, iz, count, ngllz,
                           ngllx);
          count++;
        }
      }
    }
  }

  *nglob_set = count;
  return index_mapping;
}

} // namespace adjacency_graph
} // namespace specfem

#endif
