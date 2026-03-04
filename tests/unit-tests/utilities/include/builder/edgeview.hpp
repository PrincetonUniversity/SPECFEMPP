#pragma once

#include "specfem/assembly/edge_types.hpp"
#include "specfem/element/tags.hpp"

namespace specfem::test_builder {

class EdgeView {
private:
  static constexpr specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim2;

  specfem::mesh_entity::element<dimension_tag> element;
  int nquad_element;
  std::string _edge_view_label;

public:
  std::vector<specfem::mesh_entity::edge<dimension_tag> > edges;

  EdgeView(const int &ngllz, const int &ngllx)
      : element(ngllz, ngllx), nquad_element(std::max(ngllx, ngllz)),
        _edge_view_label("specfem::test_builder::EdgeView::edgelist") {}

  specfem::assembly::EdgeView<Kokkos::DefaultExecutionSpace> build_on_device() {
    const int num_edges = edges.size();
    specfem::assembly::EdgeView<Kokkos::DefaultExecutionSpace> edgelist(
        "dshape::edgelist", num_edges, nquad_element);

    const auto h_edgelist = specfem::assembly::edge_view_from_collected_edges(
        _edge_view_label + "_host_mirror", edges, element);

    // specfem::assembly::edge_types<dimension_tag>::deep_copy(edgelist,
    // h_edgelist);
    Kokkos::deep_copy(edgelist.element_index, h_edgelist.element_index);
    Kokkos::deep_copy(edgelist.edge_index, h_edgelist.edge_index);
    Kokkos::deep_copy(edgelist.edge_types, h_edgelist.edge_types);
    Kokkos::deep_copy(edgelist.iz, h_edgelist.iz);
    Kokkos::deep_copy(edgelist.ix, h_edgelist.ix);
    return edgelist;
  }

public:
  // builder pattern
  EdgeView &add_edge(const specfem::mesh_entity::edge<dimension_tag> &edge) {
    edges.push_back(edge);
    return *this;
  }

  EdgeView &set_label(const std::string &label) {
    _edge_view_label = label;
    return *this;
  }
};

} // namespace specfem::test_builder
