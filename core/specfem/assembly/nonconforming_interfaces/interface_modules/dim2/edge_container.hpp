#pragma once

#include "../edge_container.hpp"
#include "Kokkos_Core.hpp"

namespace specfem::assembly::interface {

namespace module {

template <specfem::element::medium_tag MediumTag>
struct single_edge_container<specfem::dimension::type::dim2, MediumTag> {
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
  static constexpr specfem::element::medium_tag Medium1 = MediumTag;
  static constexpr specfem::element::medium_tag Medium2 = MediumTag;
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
  int num_edges;
  IndexView index_mapping;
  IndexView::HostMirror h_index_mapping;
  EdgeTypeView edge_type;
  EdgeTypeView::HostMirror h_edge_type;
};

template <specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2>
struct double_edge_container<specfem::dimension::type::dim2, MediumTag1,
                             MediumTag2> {
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
  static constexpr specfem::element::medium_tag Medium1 = MediumTag1;
  static constexpr specfem::element::medium_tag Medium2 = MediumTag2;
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
};
} // namespace module
} // namespace specfem::assembly::interface
