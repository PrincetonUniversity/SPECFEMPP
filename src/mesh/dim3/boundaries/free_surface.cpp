#include "mesh/mesh.hpp"
#include <iostream>

void specfem::mesh::free_surface<specfem::dimension::type::dim3>::print()
    const {

  // Print variables and metadata
  std::cout << "Absorbing Boundary Metadata:" << std::endl;
  std::cout << "================================================" << std::endl;
  std::cout << "  nelements:.............. " << nelements << std::endl;
  std::cout << "  ngllsquare:............. " << ngllsquare << std::endl;
  std::cout << "  num_free_surface_faces:. " << num_free_surface_faces
            << std::endl;

  // Print the absorbing ispec metadata
  std::cout << "  Array sizes:" << std::endl;
  std::cout << "  -----------------------------------------------" << std::endl;
  std::cout << "  ispec:.................. " << ispec.extent(0) << std::endl;
  std::cout << "  ijk:.................... " << ijk.extent(0) << " "
            << ijk.extent(1) << " " << ijk.extent(2) << std::endl;
  std::cout << "  jacobian2Dw:............ " << jacobian2Dw.extent(0) << " "
            << jacobian2Dw.extent(1) << std::endl;
  std::cout << "  normal:................. " << normal.extent(0) << " "
            << normal.extent(1) << " " << normal.extent(2) << std::endl;
}
