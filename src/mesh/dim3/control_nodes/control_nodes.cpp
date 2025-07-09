#include "mesh/dim3/control_nodes/control_nodes.hpp"
#include "iostream"

void specfem::mesh::control_nodes<specfem::dimension::type::dim3>::print()
    const {
  std::cout << "Control Nodes Information:\n"
            << "------------------------------\n"
            << "Number of spectral elements: " << nspec << "\n"
            << "Number of control nodes per spectral element: " << ngnod << "\n"
            << "Total number of distinct control nodes: "
            << coordinates.extent(0) << "\n"
            << "------------------------------\n";
}
