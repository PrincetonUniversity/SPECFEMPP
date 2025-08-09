#pragma once

#include "../edge_container.hpp"
#include "Kokkos_Core.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include <stdexcept>

namespace specfem::assembly::interface {

namespace module {

template <> struct single_edge_container<specfem::dimension::type::dim2> {
private:
  using IndexView =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< indices
  using EdgeTypeView =
      Kokkos::View<specfem::enums::edge::type *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store edge types
public:
  single_edge_container(int num_edges)
      : num_edges(num_edges),
        index_mapping("specfem::assembly::interface::module::single_edge_"
                      "container::index_mapping",
                      num_edges),
        h_index_mapping(Kokkos::create_mirror_view(index_mapping)),
        edge_type("specfem::assembly::interface::module::single_edge_"
                  "container::edge_type",
                  num_edges),
        h_edge_type(Kokkos::create_mirror_view(edge_type)) {}
  single_edge_container(const specfem::assembly::interface::initializer &init)
      : single_edge_container(init.num_medium1_edges + init.num_medium2_edges) {
    if (init.nquad_medium1 != init.nquad_medium2) {
      throw std::runtime_error(
          std::string("single_edge_container requires the same number of "
                      "quadrature points for both media. (medium 1: ") +
          std::to_string(init.nquad_medium1) + " points, medium 2: " +
          std::to_string(init.nquad_medium2) + " points)");
    }
  }
  int num_edges;
  IndexView index_mapping;
  IndexView::HostMirror h_index_mapping;
  EdgeTypeView edge_type;
  EdgeTypeView::HostMirror h_edge_type;

  template <int medium, bool access_from_host = false>
  inline auto &get_edge_index_view() {
    static_assert(medium == 1 || medium == 2, "medium must be 1 or 2!");
    if constexpr (access_from_host) {
      return index_mapping;
    } else {
      return h_index_mapping;
    }
  }
  template <int medium, bool access_from_host = false>
  inline auto &get_edge_type_view() {
    static_assert(medium == 1 || medium == 2, "medium must be 1 or 2!");
    if constexpr (access_from_host) {
      return edge_type;
    } else {
      return h_edge_type;
    }
  }
  template <specfem::sync::kind synctype> void sync_edge_container() {
    if constexpr (synctype == specfem::sync::kind::DeviceToHost) {
      Kokkos::deep_copy(h_index_mapping, index_mapping);
      Kokkos::deep_copy(h_edge_type, edge_type);
    } else {
      Kokkos::deep_copy(index_mapping, h_index_mapping);
      Kokkos::deep_copy(edge_type, h_edge_type);
    }
  }
};

template <> struct double_edge_container<specfem::dimension::type::dim2> {
private:
  using IndexView =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< indices
  using EdgeTypeView =
      Kokkos::View<specfem::enums::edge::type *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store edge types

public:
  double_edge_container(int num_medium1_edges, int num_medium2_edges)
      : num_medium1_edges(num_medium1_edges),
        num_medium2_edges(num_medium2_edges),
        medium1_index_mapping(
            "specfem::assembly::interface::module::double_edge_"
            "container::medium1_index_mapping",
            num_medium1_edges),
        h_medium1_index_mapping(
            Kokkos::create_mirror_view(medium1_index_mapping)),
        medium1_edge_type("specfem::assembly::interface::module::double_edge_"
                          "container::medium1_edge_type",
                          num_medium1_edges),
        h_medium1_edge_type(Kokkos::create_mirror_view(medium1_edge_type)),
        medium2_index_mapping(
            "specfem::assembly::interface::module::double_edge_"
            "container::medium2_index_mapping",
            num_medium2_edges),
        h_medium2_index_mapping(
            Kokkos::create_mirror_view(medium2_index_mapping)),
        medium2_edge_type("specfem::assembly::interface::module::double_edge_"
                          "container::medium2_edge_type",
                          num_medium2_edges),
        h_medium2_edge_type(Kokkos::create_mirror_view(medium2_edge_type)) {}
  double_edge_container(const specfem::assembly::interface::initializer &init)
      : double_edge_container(init.num_medium1_edges, init.num_medium2_edges) {}
  int num_medium1_edges;
  int num_medium2_edges;
  IndexView medium1_index_mapping;
  IndexView::HostMirror h_medium1_index_mapping;
  EdgeTypeView medium1_edge_type;
  EdgeTypeView::HostMirror h_medium1_edge_type;
  IndexView medium2_index_mapping;
  IndexView::HostMirror h_medium2_index_mapping;
  EdgeTypeView medium2_edge_type;
  EdgeTypeView::HostMirror h_medium2_edge_type;

  template <int medium, bool access_from_host = false>
  inline auto &get_edge_index_view() {
    static_assert(medium == 1 || medium == 2, "medium must be 1 or 2!");
    if (medium == 1) {
      if constexpr (access_from_host) {
        return medium1_index_mapping;
      } else {
        return h_medium1_index_mapping;
      }
    } else {
      if constexpr (access_from_host) {
        return medium2_index_mapping;
      } else {
        return h_medium2_index_mapping;
      }
    }
  }
  template <int medium, bool access_from_host = false>
  inline auto &get_edge_type_view() {
    static_assert(medium == 1 || medium == 2, "medium must be 1 or 2!");
    if (medium == 1) {
      if constexpr (access_from_host) {
        return medium1_edge_type;
      } else {
        return h_medium1_edge_type;
      }
    } else {
      if constexpr (access_from_host) {
        return medium2_edge_type;
      } else {
        return h_medium2_edge_type;
      }
    }
  }
  template <specfem::sync::kind synctype> void sync_edge_container() {
    if constexpr (synctype == specfem::sync::kind::DeviceToHost) {
      Kokkos::deep_copy(h_medium1_index_mapping, medium1_index_mapping);
      Kokkos::deep_copy(h_medium1_edge_type, medium1_edge_type);
      Kokkos::deep_copy(h_medium2_index_mapping, medium2_index_mapping);
      Kokkos::deep_copy(h_medium2_edge_type, medium2_edge_type);
    } else {
      Kokkos::deep_copy(medium1_index_mapping, h_medium1_index_mapping);
      Kokkos::deep_copy(medium1_edge_type, h_medium1_edge_type);
      Kokkos::deep_copy(medium2_index_mapping, h_medium2_index_mapping);
      Kokkos::deep_copy(medium2_edge_type, h_medium2_edge_type);
    }
  }
};
} // namespace module
} // namespace specfem::assembly::interface
