#include "mesh/dim3/generate_database/control_nodes/control_nodes.hpp"
#include <sstream>

std::string
specfem::mesh::control_nodes<specfem::dimension::type::dim3>::print() const {
  std::ostringstream message;
  message << "Control Nodes Information:\n"
          << "------------------------------\n"
          << "Number of spectral elements: " << nspec << "\n"
          << "Number of control nodes per spectral element: " << ngnod << "\n"
          << "Total number of distinct control nodes: " << coordinates.extent(0)
          << "\n"
          << "------------------------------\n";

  return message.str();
}
