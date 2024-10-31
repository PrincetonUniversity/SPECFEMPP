#ifndef _ADJACENCY_GRAPH_HPP_
#define _ADJACENCY_GRAPH_HPP_

#include "compute/fields/simulation_field.tpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include <vector>

namespace specfem {
namespace adjacency_graph {

struct adjacency_pointer {
  int elem;
  int8_t side;
  bool flip;

  KOKKOS_INLINE_FUNCTION
  adjacency_pointer() : elem(-1), side(0), flip(false) {}

  KOKKOS_INLINE_FUNCTION
  adjacency_pointer(int elem, int8_t side, bool flip = false)
      : elem(elem), side(side), flip(flip) {}

  KOKKOS_INLINE_FUNCTION
  adjacency_pointer(int elem, specfem::enums::boundaries::type side,
                    bool flip = false)
      : elem(elem), side(0), flip(flip) {
    switch (side) {
    case specfem::enums::boundaries::type::RIGHT:
      this->side = 0;
      break;
    case specfem::enums::boundaries::type::TOP:
      this->side = 1;
      break;
    case specfem::enums::boundaries::type::LEFT:
      this->side = 2;
      break;
    case specfem::enums::boundaries::type::BOTTOM:
      this->side = 3;
      break;
    }
  }

  KOKKOS_INLINE_FUNCTION
  ~adjacency_pointer() {}

  bool is_active() {
    // must return false for default constructor
    return elem != -1;
  }
};

template <int num_sides> struct adjacency_graph {
public:
  adjacency_graph(int nspec) : nspec(nspec) {
    adjacencies =
        Kokkos::View<specfem::adjacency_graph::adjacency_pointer *[num_sides],
                     Kokkos::LayoutLeft, specfem::kokkos::DevMemSpace>(
            "specfem::adjacency_graph::adjacency_graph::adjacencies", nspec);

    h_adjacencies =
        Kokkos::View<specfem::adjacency_graph::adjacency_pointer *[num_sides],
                     Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>(
            Kokkos::create_mirror_view(adjacencies));
    for (int ispec = 0; ispec < nspec; ispec++) {
      for (int iedge = 0; iedge < num_sides; iedge++)
        h_adjacencies(ispec, iedge).elem = -1;
    }
    Kokkos::deep_copy(adjacencies, h_adjacencies);
  }

  void clear_adjacency(int elem, int8_t side) {
    if (modified)
      Kokkos::deep_copy(h_adjacencies, adjacencies);

    specfem::adjacency_graph::adjacency_pointer neighbor =
        h_adjacencies(elem, side);
    if (neighbor.is_active()) {
      h_modified = true;
      h_adjacencies(neighbor.elem, neighbor.side) =
          specfem::adjacency_graph::adjacency_pointer();
      h_adjacencies(elem, side) = specfem::adjacency_graph::adjacency_pointer();
    }
  }

  void form_adjacency(int elem_a, int8_t side_a, int elem_b, int8_t side_b,
                      bool flip = false) {
    clear_adjacency(elem_a, side_a);
    clear_adjacency(elem_b, side_b);
    h_modified = true;
    h_adjacencies(elem_a, side_a).elem = elem_b;
    h_adjacencies(elem_a, side_a).side = side_b;
    h_adjacencies(elem_a, side_a).flip = flip;

    h_adjacencies(elem_b, side_b).elem = elem_a;
    h_adjacencies(elem_b, side_b).side = side_a;
    h_adjacencies(elem_b, side_b).flip = flip;
  }

  void form_adjacency(specfem::adjacency_graph::adjacency_pointer a,
                      specfem::adjacency_graph::adjacency_pointer b,
                      bool flip = false) {
    form_adjacency(a.elem, a.side, b.elem, b.side, flip);
  }

  specfem::adjacency_graph::adjacency_pointer get_adjacency(int elem,
                                                            int8_t side) const {
    if (modified)
      Kokkos::deep_copy(h_adjacencies, adjacencies);
    return h_adjacencies(elem, side);
  }

  int get_size() const { return nspec; }

private:
  Kokkos::View<specfem::adjacency_graph::adjacency_pointer *[num_sides],
               Kokkos::LayoutLeft, specfem::kokkos::DevMemSpace>
      adjacencies;
  Kokkos::View<specfem::adjacency_graph::adjacency_pointer *[num_sides],
               Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
      h_adjacencies;
  bool modified;
  bool h_modified;
  int nspec;
};

specfem::adjacency_graph::adjacency_graph<4> from_index_mapping(
    const Kokkos::View<int ***, Kokkos::LayoutLeft,
                       Kokkos::DefaultExecutionSpace> &index_mapping);
Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
to_index_mapping(const specfem::adjacency_graph::adjacency_graph<4> &graph,
                 const int ngllz, const int ngllx, int *nglob_set);

} // namespace adjacency_graph
} // namespace specfem

#endif
