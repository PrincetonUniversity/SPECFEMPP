#include "mesh/mesh.hpp"
#include <iostream>
#include <sstream>

std::string
specfem::mesh::free_surface<specfem::dimension::type::dim3>::print() const {

  std::ostringstream message;

  // Print variables and metadata
  message << "Absorbing Boundary Metadata:"
          << "\n";
  message << "================================================"
          << "\n";
  message << "  nelements:.............. " << nelements << "\n";
  message << "  ngllsquare:............. " << ngllsquare << "\n";
  message << "  num_free_surface_faces:. " << num_free_surface_faces << "\n";

  // Print the absorbing ispec metadata
  message << "  Array sizes:"
          << "\n";
  message << "  -----------------------------------------------"
          << "\n";
  message << "  ispec:.................. " << ispec.extent(0) << "\n";
  message << "  ijk:.................... " << ijk.extent(0) << " "
          << ijk.extent(1) << " " << ijk.extent(2) << "\n";
  message << "  jacobian2Dw:............ " << jacobian2Dw.extent(0) << " "
          << jacobian2Dw.extent(1) << "\n";
  message << "  normal:................. " << normal.extent(0) << " "
          << normal.extent(1) << " " << normal.extent(2) << "\n";

  return message.str();
}
