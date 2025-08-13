#pragma once

#include "enumerations/interface.hpp"
#include <boost/graph/adjacency_list.hpp>

namespace specfem::mesh {

template <specfem::dimension::type Dimension> struct adjacency_graph {

public:
  struct EdgeProperties {
    specfem::connections::type connection;
    specfem::mesh_entity::type orientation;

    EdgeProperties() = default;

    EdgeProperties(const specfem::connections::type conn,
                   const specfem::mesh_entity::type orient)
        : connection(conn), orientation(orient) {}
  };

private:
  using Graph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                            boost::no_property, EdgeProperties>;
  Graph graph_;

public:
  adjacency_graph() = default;

  adjacency_graph(const int nspec) : graph_(nspec) {}

  Graph &graph() { return graph_; }

  const Graph &graph() const { return graph_; }

  const bool empty() const { return boost::num_vertices(graph_) == 0; }
};

} // namespace specfem::mesh
