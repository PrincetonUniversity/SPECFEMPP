#pragma once

#include "locate_point_impl.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <vector>

namespace {
template <typename GraphType>
std::vector<int> get_best_candidates_from_graph(const int ispec_guess,
                                                const GraphType &graph) {

  std::vector<int> ispec_candidates;
  ispec_candidates.push_back(ispec_guess);

  for (auto edge :
       boost::make_iterator_range(boost::out_edges(ispec_guess, graph))) {
    const int ispec = boost::target(edge, graph);
    if (std::find(ispec_candidates.begin(), ispec_candidates.end(), ispec) ==
        ispec_candidates.end()) {
      ispec_candidates.push_back(ispec);
    }
  }
  return ispec_candidates;
}
} // namespace

namespace specfem::algorithms::locate_point_impl {

template <typename GraphType>
specfem::point::local_coordinates<specfem::dimension::type::dim2>
locate_point_core(
    const GraphType &graph,
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const Kokkos::View<
        type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
        &global_coordinates,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &control_node_coord,
    const int ngnod) {

  int ix_guess, iz_guess, ispec_guess;

  std::tie(ix_guess, iz_guess, ispec_guess) =
      rough_location(coordinates, global_coordinates);

  const auto best_candidates =
      get_best_candidates_from_graph(ispec_guess, graph);

  return locate_point_from_best_candidates(best_candidates, coordinates,
                                           control_node_coord, ngnod);
}
} // namespace specfem::algorithms::locate_point_impl
