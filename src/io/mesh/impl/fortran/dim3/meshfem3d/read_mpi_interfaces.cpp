#include "io/mesh/impl/fortran/dim3/meshfem3d/read_mpi_interfaces.hpp"
#include "io/fortranio/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include <fstream>

void specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_mpi_interfaces(
    std::ifstream &stream, const specfem::MPI::MPI &mpi) {

  int num_interfaces, max_elements_per_interface;

  specfem::io::fortran_read_line(stream, &num_interfaces,
                                 &max_elements_per_interface);

  // TODO (Rohit: MPI_INTERFACES): Add support for MPI interfaces
  if (num_interfaces != 0 && max_elements_per_interface != 0) {
    throw std::runtime_error(
        "MPI interfaces are not supported yet for 3D simulations.");
  }
}
