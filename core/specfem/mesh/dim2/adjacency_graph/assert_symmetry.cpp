#include "adjacency_graph.hpp"
#include <boost/range/iterator_range.hpp>
#include <sstream>
#include <stdexcept>

template <>
void specfem::mesh::adjacency_graph<
    specfem::dimension::type::dim2>::assert_symmetry() const {
  const auto &g = this->graph();

  for (const auto &edge : boost::make_iterator_range(boost::edges(g))) {
    const auto source = boost::source(edge, g);
    const auto target = boost::target(edge, g);
    if (!boost::edge(target, source, g).second) {
      std::ostringstream message;
      message << "Adjacency graph is not symmetric: edge from " << source
              << " to " << target << " exists, but not from " << target
              << " to " << source;
      throw std::runtime_error(message.str());
    }
  }
  return;
}
