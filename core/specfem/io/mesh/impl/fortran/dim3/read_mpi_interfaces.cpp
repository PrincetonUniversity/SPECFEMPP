#include "specfem/io/mesh/impl/fortran/dim3/read_mpi_interfaces.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include <fstream>

void specfem::io::mesh::impl::fortran::dim3::read_mpi_interfaces(
    std::ifstream &stream) {

  int num_interfaces, max_elements_per_interface;

  specfem::io::fortran_read_line(stream, &num_interfaces,
                                 &max_elements_per_interface);

  // Legacy MPI interfaces section: MPI connectivity is now handled entirely
  // by the adjacency graph (local and MPI adjacencies). This section always
  // contains 0, 0 to maintain database format compatibility.
  if (num_interfaces != 0 || max_elements_per_interface != 0) {
    throw std::runtime_error(
        "Unexpected non-zero MPI interface counts. Database format mismatch. "
        "MPI interfaces are now exclusively handled by the adjacency graph.");
  }
}
