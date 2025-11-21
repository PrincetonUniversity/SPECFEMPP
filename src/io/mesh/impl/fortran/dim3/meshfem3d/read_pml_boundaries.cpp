#include "io/mesh/impl/fortran/dim3/meshfem3d/read_pml_boundaries.hpp"
#include "io/fortranio/interface.hpp"

#include <fstream>

void specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_pml_boundaries(
    std::ifstream &stream) {

  int num_pml_boundaries;

  specfem::io::fortran_read_line(stream, &num_pml_boundaries);

  // TODO (Rohit: PML_BOUNDARIES): Add support for PML boundaries
  if (num_pml_boundaries != 0) {
    throw std::runtime_error(
        "PML boundaries are not supported yet for 3D simulations.");
  }
}
