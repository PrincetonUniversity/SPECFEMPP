#include "adjacency_graph.hpp"
#include <boost/range/iterator_range.hpp>
#include <sstream>
#include <stdexcept>

void specfem::mesh::adjacency_graph<
    specfem::dimension::type::dim3>::assert_symmetry() const {
  const auto &g = this->graph();

  for (const auto &edge : boost::make_iterator_range(boost::edges(g))) {
    const auto source = boost::source(edge, g);
    const auto target = boost::target(edge, g);
    const auto &edge_props = g[edge];
    const auto target_edge =
        boost::edge(target, source, g); // Check for reverse edge
    if (!target_edge.second) {
      // Reverse edge does not exist, graph is not symmetric
      std::ostringstream message;
      message << "Adjacency graph is not symmetric: edge from " << source
              << " to " << target << " exists, but not from " << target
              << " to " << source;
      throw std::runtime_error(message.str());
    }
    const auto &target_edge_props = g[target_edge.first];
    if (edge_props.connection != target_edge_props.connection) {
      // Connection types do not match, graph is not symmetric
      std::ostringstream message;
      message << "Adjacency graph is not symmetric: edge from " << source
              << " to " << target << " has connection type "
              << specfem::connections::to_string(edge_props.connection)
              << ", but reverse edge has connection type "
              << specfem::connections::to_string(target_edge_props.connection);
      throw std::runtime_error(message.str());
    }
  }
  return;
}
